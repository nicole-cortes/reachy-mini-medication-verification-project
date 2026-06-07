"""Verify a dose: the user shows pills, we check them against a stored medication.

The LLM should call this when the user says something like
"can you check if these are right" or "I'm about to take my morning meds".
"""

from __future__ import annotations

import logging
from typing import Any

import cv2

from reachy_medication_app.medication.pipeline import capture_multiview_frames, tray_scan_views
from reachy_medication_app.medication.gemini_vlm import call_gemini
from reachy_medication_app.medication.store import get, list_all, log_dose_taken
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


_SMALL_NUMBERS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _to_int_or_none(value: Any) -> Any:
    """Coerce a model-supplied value to int, or None if it isn't a clean number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_difference(pills_seen: Any, expected_count: Any) -> Any:
    """Return (seen - expected) as an int, or None if either isn't a clean number."""
    try:
        return int(pills_seen) - int(expected_count)
    except (TypeError, ValueError):
        return None


def _count_delta_phrase(diff: Any) -> str:
    """Turn a numeric count difference into a short spoken phrase.

    diff > 0 means extra pills; diff < 0 means missing pills.
    """
    if not isinstance(diff, int) or diff == 0:
        return ""
    n = abs(diff)
    word = _SMALL_NUMBERS.get(n, str(n))
    pill = "pill" if n == 1 else "pills"
    verb = "appears" if n == 1 else "appear"
    if diff > 0:
        return f"There {verb} to be {word} extra {pill}."
    return f"There {verb} to be {word} {pill} missing."


def _spoken_verification_summary(
    *,
    drug_name: str,
    safe: bool,
    verdict: str,
    pills_seen: Any,
    pills_described: str,
    explanation: str,
    confidence: str,
    expected_count: Any = None,
    nonmatching_pills: Any = None,
) -> str:
    if safe:
        detail = pills_described.strip() or f"{pills_seen} pill{'s' if pills_seen != 1 else ''}"
        return f"That looks right. I see {detail} for your {drug_name}."

    delta = _count_delta_phrase(_count_difference(pills_seen, expected_count))

    # Foreign pills in the tray — flag them even if one correct pill is present.
    try:
        foreign = int(nonmatching_pills)
    except (TypeError, ValueError):
        foreign = None
    if verdict in ("foreign_pills_present", "mismatch_both") or (foreign is not None and foreign > 0):
        seen_part = (
            f"I see {pills_seen} pills in the tray"
            if pills_seen not in (None, "")
            else "I see more than one kind of pill in the tray"
        )
        if foreign and foreign > 0:
            n = abs(foreign)
            word = _SMALL_NUMBERS.get(n, str(n))
            dont = f"{word} {'pill does' if n == 1 else 'pills do'} not"
            return (
                f"{seen_part}, but {dont} match your {drug_name}. "
                f"Please take out anything that isn't your {drug_name} and let me check again "
                f"before you take it."
            )
        return (
            f"{seen_part}, and some of them do not match your {drug_name}. "
            f"Please remove anything that isn't your {drug_name} and let me check again."
        )

    if verdict == "mismatch_count" and pills_seen not in (None, ""):
        base = (
            f"I see {pills_seen} pill{'s' if pills_seen != 1 else ''}, so that does not "
            f"match your saved dose of {expected_count}."
            if expected_count not in (None, "")
            else f"I see {pills_seen} pill{'s' if pills_seen != 1 else ''}, so the count does not match {drug_name}."
        )
        tail = delta or "Please double-check before taking it."
        return f"{base} {tail}".strip()
    if verdict == "mismatch_appearance":
        return f"These do not look like your {drug_name}. Please double-check the bottle before taking anything."
    if verdict == "mismatch_both":
        count_part = f" {delta}" if delta else ""
        return (
            f"The count and appearance do not match your {drug_name}.{count_part} "
            f"Please double-check the bottle before taking anything."
        ).strip()

    detail = explanation.strip() or "I could not verify the pills clearly."
    if confidence == "low":
        return f"I'm not fully sure yet. {detail} Please spread the pills out and let me try again."
    return f"I couldn't verify these safely. {detail}"


def _build_prompt(med_description: str, expected_count: int, color: str, shape: str, appearance: str) -> str:
    """Build a verification prompt that tells Gemini what to look for."""
    return f"""You are helping an older adult verify they have the correct medication in front of them BEFORE they take it. Patient safety is critical — when in doubt, say so clearly.

You are given multiple views of the same tray. Use the clearest view or combine views when counting.

For this dose the user should have EXACTLY {expected_count} pill(s), and EVERY pill in the tray should match this medication:
  Medication: {med_description}
  Color: {color or 'unspecified'}
  Shape: {shape or 'unspecified'}
  Appearance: {appearance or 'unspecified'}

Do all of the following:
1. Count the TOTAL number of pills visible anywhere in the tray — every single pill, including ones that look different from the medication above. Do NOT count only the matching ones.
2. Of those, count how many clearly MATCH the medication description.
3. Count how many do NOT match (a different color, shape, or size means it is a different/foreign pill).

Return ONLY valid JSON (no markdown, no extra text):
{{
  "total_pills_seen": <integer, every pill in the tray>,
  "matching_pills": <integer, how many match the expected medication>,
  "nonmatching_pills": <integer, how many do NOT match the expected medication>,
  "pills_described": "<short description of everything you see, e.g. '1 white round tablet and 5 assorted colored pills'>",
  "count_matches": <true if total_pills_seen == {expected_count}>,
  "all_pills_match": <true ONLY if every visible pill matches the expected medication and there are no foreign pills>,
  "appearance_matches": <true if the matching pills look right, false if they look different, "unsure" if you can't tell>,
  "verdict": "<one of: 'match', 'mismatch_count', 'foreign_pills_present', 'mismatch_appearance', 'mismatch_both', 'unclear'>",
  "confidence": "<one of: 'high', 'medium', 'low'>",
  "explanation": "<one short sentence explaining the verdict>",
  "safe_to_proceed": <true ONLY if total_pills_seen == {expected_count} AND all_pills_match is true AND confidence is 'high' or 'medium'>
}}

Rules:
- safe_to_proceed must be FALSE if there are ANY pills that do not match the expected medication, even if a correct pill is also present.
- safe_to_proceed must be FALSE if total_pills_seen is not exactly {expected_count}.
- If there are foreign/non-matching pills, set verdict to 'foreign_pills_present' (or 'mismatch_both' if the count is also wrong).
- Count EVERY pill in the tray for total_pills_seen — never report only the matching pills.
- If you cannot see the pills clearly, set verdict to 'unclear' and safe_to_proceed to false.
- If pills are overlapping or you're unsure of the count, prefer 'unclear' over guessing."""


class VerifyDose(Tool):
    """Check that the pills the user is about to take match a stored medication."""

    name = "verify_dose"
    description = (
        "Verify the pills the user is currently showing match a saved medication. "
        "Use this BEFORE the user takes a dose, when they say 'check this', "
        "'is this right', 'am I taking the right medicine', or similar. You MUST "
        "pass the drug_name of the medication being verified — ask the user if you "
        "don't know which one. This tool first looks down at the pill tray before "
        "capturing the image. If safe_to_proceed is false, tell the user the "
        "specific reason and recommend they double-check the bottle or ask their "
        "pharmacist before taking anything."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "The drug name to verify against. Must match a medication previously added via add_medication.",
            },
            "log_if_safe": {
                "type": "boolean",
                "description": "If true and the verification passes, log the dose as taken. Defaults to false; only set true when the user has confirmed they are taking the dose now.",
                "default": False,
            },
        },
        "required": ["drug_name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        drug_name = str(kwargs.get("drug_name", "")).strip()
        log_if_safe = bool(kwargs.get("log_if_safe", False))
        logger.info("Tool call: verify_dose drug_name=%s log_if_safe=%s", drug_name, log_if_safe)

        if not drug_name:
            stored = list_all()
            if not stored:
                return {
                    "status": "error",
                    "error": "no_drug_name_and_no_meds",
                    "spoken_summary": "I don't have any medications saved yet.",
                    "followup_question": "Would you like me to read a bottle and add one?",
                    "message_to_user_hint": (
                        "Tell the user you don't have any medications saved yet, "
                        "and offer to read a bottle to add one."
                    ),
                }
            return {
                "status": "needs_clarification",
                "stored_drug_names": [m.drug_name for m in stored],
                "spoken_summary": "I need to know which saved medication you want me to check.",
                "message_to_user_hint": (
                    "Ask the user which medication they're verifying — read off the "
                    "names you have stored."
                ),
            }

        med = get(drug_name)
        if med is None:
            return {
                "status": "error",
                "error": "drug_not_found",
                "requested": drug_name,
                "stored_drug_names": [m.drug_name for m in list_all()],
                "spoken_summary": f"I don't have {drug_name} saved yet.",
                "message_to_user_hint": (
                    f"Tell the user you don't have {drug_name} saved yet. List the "
                    f"medications you DO know about and offer to read a new bottle "
                    f"if needed."
                ),
            }

        # Capture image
        try:
            frames = capture_multiview_frames(deps, tray_scan_views())
        except Exception as e:
            logger.exception("verify_dose: capture failed")
            return {
                "status": "error",
                "error": f"capture_failed: {e}",
                "spoken_summary": (
                    "I couldn't see the pills clearly. Please show them again with good lighting."
                ),
                "message_to_user_hint": (
                    "Tell the user you couldn't see clearly and ask them to show the "
                    "pills again with good lighting."
                ),
            }

        jpeg_images: list[bytes] = []
        for frame in frames:
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                jpeg_images.append(enc.tobytes())

        if not jpeg_images:
            return {
                "status": "error",
                "error": "jpeg_encode_failed",
                "spoken_summary": "Something went wrong while preparing the tray images. Please try again.",
                "message_to_user_hint": "Tell the user something went wrong on your end and ask them to try once more.",
            }

        prompt = _build_prompt(
            med_description=f"{med.drug_name} {med.strength}",
            expected_count=med.dosage_count,
            color=med.color,
            shape=med.shape,
            appearance=med.appearance,
        )
        result = call_gemini(prompt, jpeg_images)
        if result["error"]:
            return {
                "status": "error",
                "error": result["error"],
                "spoken_summary": (
                    "I had trouble with the vision check. Please confirm with the bottle before taking anything."
                ),
                "message_to_user_hint": "Tell the user the vision check had trouble and recommend they confirm with the bottle before taking anything.",
            }

        parsed = result["parsed_json"]
        if not parsed:
            return {
                "status": "error",
                "error": "could_not_parse",
                "raw_text": result["raw_text"],
                "spoken_summary": (
                    "I got a confusing answer from the image, so please double-check before taking the dose."
                ),
                "message_to_user_hint": "Tell the user you got a confusing answer and recommend they double-check before taking the dose.",
            }

        verdict = parsed.get("verdict", "unclear")
        confidence = parsed.get("confidence", "low")

        # Total pills in the tray. Accept the new field name, fall back to the
        # old one for safety if the model returns the legacy shape.
        pills_seen = parsed.get("total_pills_seen", parsed.get("pills_seen"))
        nonmatching_pills = _to_int_or_none(parsed.get("nonmatching_pills"))
        all_pills_match = parsed.get("all_pills_match")
        count_difference = _count_difference(pills_seen, med.dosage_count)

        # Code-side safety guard: never report "safe" if the count is off or any
        # foreign pills are present, regardless of what the model claimed. This
        # prevents a false "looks right" when only one correct pill is present
        # alongside others.
        model_safe = bool(parsed.get("safe_to_proceed", False))
        count_ok = _count_difference(pills_seen, med.dosage_count) == 0
        no_foreign = (nonmatching_pills in (None, 0)) and (all_pills_match is not False)
        safe = model_safe and count_ok and no_foreign

        # If the model said safe but our guard overrode it, make the verdict honest.
        if model_safe and not safe and verdict in ("match", "unclear"):
            if not count_ok:
                verdict = "foreign_pills_present" if (nonmatching_pills or 0) > 0 else "mismatch_count"
            elif not no_foreign:
                verdict = "foreign_pills_present"

        logged = None
        if log_if_safe and safe:
            logged = log_dose_taken(med.drug_name, notes=parsed.get("explanation", ""))

        return {
            "status": "ok",
            "drug_name": med.drug_name,
            "expected_count": med.dosage_count,
            "expected_appearance": med.appearance or med.color,
            "pills_seen": pills_seen,
            "matching_pills": _to_int_or_none(parsed.get("matching_pills")),
            "nonmatching_pills": nonmatching_pills,
            "count_difference": count_difference,
            "pills_described": parsed.get("pills_described", ""),
            "verdict": verdict,
            "confidence": confidence,
            "safe_to_proceed": safe,
            "explanation": parsed.get("explanation", ""),
            "dose_logged": logged is not None,
            "spoken_summary": _spoken_verification_summary(
                drug_name=med.drug_name,
                safe=safe,
                verdict=verdict,
                pills_seen=pills_seen,
                pills_described=parsed.get("pills_described", ""),
                explanation=parsed.get("explanation", ""),
                confidence=confidence,
                expected_count=med.dosage_count,
                nonmatching_pills=nonmatching_pills,
            ),
            "message_to_user_hint": (
                "If safe_to_proceed is TRUE: confirm naturally, like 'That looks right — "
                "go ahead and take it.' If FALSE: tell the user the specific reason. If "
                "nonmatching_pills is greater than 0, there are foreign pills in the tray — "
                "tell the user some pills don't match their medication and ask them to remove "
                "anything that isn't this medication, then re-check. For a count mismatch, "
                "state how many pills you see versus the saved dose (see count_difference: "
                "positive means extra, negative means missing). Never tell them it's fine if "
                "safe_to_proceed is false."
            ),
        }

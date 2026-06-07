"""Bottle-scanning tool: read a medication label via Gemini and return structured info.

Called when the user shows a new pill bottle and says something like
"add this medication" or "scan this bottle". Returns extracted fields so
the LLM can read them back to the user for confirmation, then the LLM
calls `add_medication` to actually save it.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2

from reachy_medication_app.medication.pipeline import (
    bottle_scan_views,
    capture_multiview_frames,
)
from reachy_medication_app.medication.gemini_vlm import call_gemini
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Prompt asks Gemini to extract every label field we care about, AND to
# self-report confidence + uncertainty so the LLM can ask follow-ups.
SCAN_BOTTLE_PROMPT = """You are helping an older adult identify a medication from a pill bottle photo.

You are given multiple views of the same bottle. Use the clearest view or combine information across views. If the label is still unclear, say so.

Look carefully at the bottle label in the image. Extract the following fields. If a field is not visible or unclear, say "unknown" — DO NOT guess.

Return ONLY valid JSON with this exact shape (no markdown fences, no extra text):
{
  "drug_name": "<the medication name, e.g. Lisinopril>",
  "strength": "<dose strength incl. units, e.g. 10mg or 500mg>",
  "instructions": "<the full label instructions verbatim, e.g. 'Take 1 tablet by mouth once daily with food'>",
  "dosage_count": <integer, how many pills per single dose, e.g. 1>,
  "frequency": "<how often per day, e.g. 'once daily', 'twice daily', 'every 8 hours'>",
  "appearance": "<short description of the pills if visible, e.g. 'small white round tablet'>",
  "color": "<dominant pill color if visible, lowercase, e.g. 'white'>",
  "shape": "<pill shape if visible: 'round', 'oval', 'capsule', 'unknown'>",
  "label_clear": <true if you can read the label confidently, false otherwise>,
  "fields_uncertain": [<list of any field names you marked 'unknown' or guessed at>],
  "notes": "<one short sentence describing anything notable: glare, blur, partial occlusion, etc.>"
}

Important:
- The drug_name and strength are the most important fields — if you can't read them, set label_clear to false and put them in fields_uncertain.
- Use 'unknown' for any field you cannot read confidently. Never invent dosage info.
- If the image is too blurry or dark to read, set label_clear to false and explain in notes."""


class ScanBottle(Tool):
    """Capture an image of a pill bottle and extract label info via Gemini."""

    name = "scan_bottle"
    description = (
        "Fallback medication-label reader that performs a stricter multi-view Gemini "
        "pass after normal `camera` reading was still unclear. Use this only when the "
        "user is still showing the bottle and you want one more structured extraction "
        "attempt for drug name, strength, instructions, dosage, frequency, and appearance."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        logger.info("Tool call: scan_bottle")

        # Capture a frame from Reachy's camera (same helper the existing
        # check_medications tool uses, so behavior is consistent).
        try:
            frames = capture_multiview_frames(deps, bottle_scan_views())
        except Exception as e:
            logger.exception("scan_bottle: failed to capture frame")
            return {
                "status": "error",
                "error": f"Could not capture image from camera: {e}",
                "message": "I couldn't get an image from my camera. Could you check that I'm connected and try again?",
            }

        # Encode to JPEG for Gemini
        jpeg_images: list[bytes] = []
        for frame in frames:
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                jpeg_images.append(enc.tobytes())

        if not jpeg_images:
            return {
                "status": "error",
                "error": "JPEG encode failed",
                "spoken_summary": "I had trouble preparing the bottle image. Let's try once more.",
                "message": "I had trouble processing the bottle images. Let's try once more.",
            }

        # Call Gemini
        result = call_gemini(SCAN_BOTTLE_PROMPT, jpeg_images)
        if result["error"]:
            return {
                "status": "error",
                "error": result["error"],
                "spoken_summary": "I had trouble analyzing the bottle. Could you hold it closer or in better light and we can try again?",
                "message": "I had trouble analyzing the bottle. Could you check the lighting or hold the bottle closer and we can try again?",
            }

        parsed = result["parsed_json"]
        if not parsed:
            return {
                "status": "error",
                "error": "Could not parse model response as JSON",
                "raw_text": result["raw_text"],
                "spoken_summary": "I saw the bottle, but I couldn't make sense of the label. Could you move it closer and try again?",
                "message": "I saw the bottle but I'm having trouble making sense of it. Could you move it closer or to better light and try again?",
            }

        # Surface the most important fields plus a friendly summary for the LLM
        drug_name = parsed.get("drug_name", "unknown")
        strength = parsed.get("strength", "unknown")
        instructions = parsed.get("instructions", "unknown")
        label_clear = parsed.get("label_clear", False)
        uncertain = parsed.get("fields_uncertain", []) or []

        return {
            "status": "ok",
            "drug_name": drug_name,
            "strength": strength,
            "instructions": instructions,
            "dosage_count": parsed.get("dosage_count"),
            "frequency": parsed.get("frequency", "unknown"),
            "appearance": parsed.get("appearance", "unknown"),
            "color": parsed.get("color", "unknown"),
            "shape": parsed.get("shape", "unknown"),
            "label_clear": label_clear,
            "fields_uncertain": uncertain,
            "notes": parsed.get("notes", ""),
            "spoken_summary": (
                f"I think this is {drug_name} {strength}. {instructions}"
                if label_clear
                else "I still couldn't read the bottle clearly. Could you hold it closer, steadier, or in better light?"
            ),
            "followup_question": (
                "Is that right?"
                if label_clear
                else ""
            ),
            "message_to_user_hint": (
                "If label_clear is true, read back the drug name, strength, and "
                "instructions in short sentences and ask the user to confirm. If "
                "label_clear is false, say you could not read the bottle clearly "
                "and ask the user to hold it closer, steadier, or in better light."
            ),
            "next_step_hint": (
                "Read the drug name, strength, and instructions back to the user. "
                "Ask them to confirm. If they say 'yes' or correct any details, call "
                "add_medication with the final values (including any doctor_override "
                "if the user said their doctor changed the instructions). "
                "If label_clear is false, ask the user to reposition the bottle and call scan_bottle again."
            ),
        }

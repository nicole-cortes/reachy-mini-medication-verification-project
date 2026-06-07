"""Check whether a medication was already taken (logged) today.

Called BEFORE the user takes a saved medication, when they say things like
"I need to take my lisinopril" or "did I already take my pills today?". This
inspects the existing dose log in the medication store (the same log that
`verify_dose` writes to when a dose is confirmed) and tells the user whether
they have already logged that medication today, and when.

This is a safety/double-dosing guard for the demo — it does NOT give clinical
advice; it only reports what was logged.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from reachy_medication_app.medication.store import doses_today, get, list_all
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _friendly_time(taken_at: Any, fallback_str: str = "") -> str:
    """Format a dose timestamp as e.g. '9:14 AM'."""
    if isinstance(taken_at, (int, float)) and taken_at > 0:
        formatted = time.strftime("%I:%M %p", time.localtime(taken_at))
        return formatted.lstrip("0")
    # Fallback to the stored "%Y-%m-%d %H:%M:%S" string if present.
    return fallback_str.strip()


class CheckDoseHistory(Tool):
    """Report whether a saved medication was already logged as taken today."""

    name = "check_dose_history"
    description = (
        "Check whether a medication has already been logged as taken today, and "
        "if so, when it was last taken. Call this when the user says things like "
        "'did I already take my lisinopril today?', 'I need to take my "
        "medication', or 'can I take this now?' — especially if they seem about to "
        "take a saved medication again. This is a double-dosing safety check, not "
        "medical advice. Pass the drug_name to check."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": (
                    "The medication to check, e.g. 'Lisinopril'. If omitted and "
                    "exactly one medication is saved, that one is used."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        drug_name = str(kwargs.get("drug_name", "")).strip()
        logger.info("Tool call: check_dose_history drug_name=%s", drug_name)

        # Resolve the medication. If no name given, use the only saved med if there's one.
        if not drug_name:
            meds = list_all()
            if len(meds) == 1:
                drug_name = meds[0].drug_name
            elif not meds:
                return {
                    "status": "error",
                    "error": "no_meds_saved",
                    "spoken_summary": "I don't have any medications saved yet.",
                    "message_to_user_hint": (
                        "Tell the user you don't have any medications saved yet, and "
                        "offer to read a bottle to add one."
                    ),
                }
            else:
                return {
                    "status": "needs_clarification",
                    "stored_drug_names": [m.drug_name for m in meds],
                    "spoken_summary": "Which medication do you mean?",
                    "message_to_user_hint": (
                        "Ask the user which saved medication they're asking about — "
                        "read off the names you have stored."
                    ),
                }

        # Use the canonical stored name for nicer phrasing and for a more reliable
        # history match, if we have it.
        med = get(drug_name)
        display_name = med.drug_name if med is not None else drug_name
        lookup_name = med.drug_name if med is not None else drug_name

        entries = doses_today(lookup_name)
        times_taken_today = len(entries)
        taken_today = times_taken_today > 0
        last_entry = entries[-1] if entries else None
        last_taken_at = (
            _friendly_time(
                last_entry.get("taken_at"),
                str(last_entry.get("taken_at_str", "")),
            )
            if last_entry is not None
            else None
        )

        if not taken_today:
            spoken_summary = (
                f"I don't have a record of you taking {display_name} today."
            )
            hint = (
                f"Tell the user you have no record of {display_name} being taken "
                f"today. You can offer to verify the pills before they take it."
            )
        else:
            if times_taken_today == 1:
                spoken_summary = (
                    f"You already logged {display_name} as taken today at "
                    f"{last_taken_at}. Please double-check before taking another dose."
                )
            else:
                spoken_summary = (
                    f"You've already logged {display_name} {times_taken_today} times "
                    f"today, most recently at {last_taken_at}. Please double-check "
                    f"before taking another dose."
                )
            hint = (
                f"Warn the user gently but clearly that {display_name} was already "
                f"logged today at {last_taken_at}. Tell them to double-check before "
                f"taking another dose. Do not tell them it is safe to take more — "
                f"that's a question for their doctor or pharmacist."
            )

        return {
            "status": "ok",
            "drug_name": display_name,
            "taken_today": taken_today,
            "times_taken_today": times_taken_today,
            "last_taken_at": last_taken_at,
            "spoken_summary": spoken_summary,
            "message_to_user_hint": hint,
        }

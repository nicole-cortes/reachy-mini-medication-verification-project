"""List the medications Reachy currently has saved."""

from __future__ import annotations

import logging
from typing import Any

from reachy_medication_app.medication.store import list_all
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ListMedications(Tool):
    """Return the list of medications Reachy is currently tracking."""

    name = "list_medications"
    description = (
        "Return the list of medications the user has added to Reachy's memory, "
        "with each medication's strength, instructions, and frequency. Use this "
        "when the user asks 'what medications do you know about', 'what's on my "
        "schedule', or 'what should I take today'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        logger.info("Tool call: list_medications")

        meds = list_all()
        if not meds:
            return {
                "status": "ok",
                "count": 0,
                "medications": [],
                "spoken_summary": "I don't have any medications saved yet.",
                "followup_question": "Would you like me to read a bottle and add one?",
                "message_to_user_hint": (
                    "Tell the user that no medications are saved yet, and offer to "
                    "read a bottle to add one."
                ),
            }

        return {
            "status": "ok",
            "count": len(meds),
            "medications": [m.to_dict() for m in meds],
            "spoken_summary": (
                "I know about "
                + ", ".join(
                    f"{med.drug_name} {med.strength}".strip()
                    for med in meds[:3]
                )
                + ("." if len(meds) <= 3 else ", and more.")
            ),
            "message_to_user_hint": (
                "Read the list naturally — say each drug name, strength, and how "
                "often to take it. Don't list everything at once if there are many; "
                "summarize and offer to read specific ones."
            ),
        }

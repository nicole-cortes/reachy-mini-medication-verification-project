"""Save a medication to Reachy's in-memory store after the user confirms.

Called by the LLM AFTER the medication details were read back and the user
verbally confirmed (or corrected) them. The LLM should pass the *final*
values to use — not necessarily the first visual read, because the user may
correct something ("my doctor said twice a day, not once").
"""

from __future__ import annotations

import logging
from typing import Any

from reachy_medication_app.medication.store import Medication, add_or_update, get
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class AddMedication(Tool):
    """Save a medication to memory after the user has confirmed the details."""

    name = "add_medication"
    description = (
        "Save a medication to Reachy's memory after the user has confirmed the "
        "details out loud. Use this AFTER the user has confirmed or corrected the "
        "medication details you read back, whether they came from `camera`, "
        "`scan_bottle`, or the user's own verbal correction. If the user says "
        "their doctor told them something different from the label, pass the "
        "doctor's instructions as `doctor_override` and keep the original label "
        "text in `label_original_instructions`."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Medication name, e.g. 'Lisinopril'.",
            },
            "strength": {
                "type": "string",
                "description": "Dose strength with units, e.g. '10mg'. Use 'unknown' if still unclear.",
            },
            "instructions": {
                "type": "string",
                "description": "Final instructions the user is going to follow. If a doctor said something different from the label, this is the doctor's instructions. Use a short placeholder like 'not confirmed yet' only if the user asked you to save before everything was confirmed.",
            },
            "dosage_count": {
                "type": "integer",
                "description": "Number of pills per dose, e.g. 1. If unknown, omit and it will default to 1 until corrected.",
            },
            "frequency": {
                "type": "string",
                "description": "How often, e.g. 'once daily', 'twice daily'. If the schedule is not known yet, use 'unspecified'.",
            },
            "appearance": {
                "type": "string",
                "description": "Short description of the pill, e.g. 'small white round tablet'.",
            },
            "color": {
                "type": "string",
                "description": "Dominant pill color, e.g. 'white'.",
            },
            "shape": {
                "type": "string",
                "description": "Pill shape: 'round', 'oval', 'capsule', etc.",
            },
            "doctor_override": {
                "type": "string",
                "description": "If the user said their doctor changed the dosing, put the doctor's instruction here. Otherwise omit.",
            },
            "label_original_instructions": {
                "type": "string",
                "description": "If doctor_override is set, put the original label text here so we have a record.",
            },
        },
        "required": ["drug_name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        logger.info("Tool call: add_medication kwargs=%s", kwargs)

        drug_name = str(kwargs.get("drug_name", "")).strip()
        if not drug_name:
            return {"status": "error", "error": "drug_name is required"}

        # Merge with any existing record so a partial update (e.g. adding pill
        # appearance later) doesn't wipe fields the user already confirmed.
        existing = get(drug_name)

        def _str_field(key: str, attr: str, default: str) -> str:
            raw = kwargs.get(key)
            if raw not in (None, ""):
                return str(raw).strip()
            if existing is not None:
                return getattr(existing, attr)
            return default

        med = Medication(
            drug_name=(existing.drug_name if existing is not None else drug_name),
            strength=_str_field("strength", "strength", "unknown"),
            instructions=_str_field("instructions", "instructions", "not confirmed yet"),
            dosage_count=int(
                kwargs.get("dosage_count")
                or (existing.dosage_count if existing is not None else 1)
            ),
            frequency=_str_field("frequency", "frequency", "unspecified"),
            appearance=_str_field("appearance", "appearance", ""),
            color=_str_field("color", "color", "").lower(),
            shape=_str_field("shape", "shape", "").lower(),
            doctor_override=(
                str(kwargs["doctor_override"]).strip()
                if kwargs.get("doctor_override")
                else (existing.doctor_override if existing is not None else None)
            ),
            label_original_instructions=_str_field(
                "label_original_instructions", "label_original_instructions", ""
            ),
        )
        saved = add_or_update(med)

        return {
            "status": "ok",
            "saved": saved.to_dict(),
            "spoken_summary": (
                f"I saved {saved.drug_name} {saved.strength}."
                if not saved.doctor_override
                else f"I saved {saved.drug_name} {saved.strength} and noted the doctor's updated instructions."
            ),
            "followup_question": "Would you like me to set a reminder for it?",
            "message_to_user_hint": (
                f"Confirm to the user that you've saved {saved.drug_name} {saved.strength} "
                f"and are ready to help them take it. If a doctor_override was set, "
                f"mention that you noted the doctor's instructions instead of the label's."
            ),
        }

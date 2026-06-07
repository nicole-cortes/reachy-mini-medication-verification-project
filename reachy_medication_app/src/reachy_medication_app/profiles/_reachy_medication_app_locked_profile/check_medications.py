"""Medication-check tool for the locked Reachy medication profile."""

from __future__ import annotations

import logging
from typing import Any

from reachy_medication_app.medication.pipeline import run_medication_check
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CheckMedications(Tool):
    """Tilt toward the medication area, capture an image, and analyze it."""

    name = "check_medications"
    description = (
        "Look down at the medication tray, take a picture, estimate visible pills "
        "and containers, and run a second verification pass. Use this when the "
        "user asks broad pill-counting questions like 'how many pills are here', "
        "'count these pills', or 'look at my tray' without naming a saved medication."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What medication-related task to perform, for example 'check my medications for today'.",
            },
            "use_openai": {
                "type": "boolean",
                "description": "Whether to use OpenAI image analysis if an API key is configured.",
                "default": True,
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        task = str(kwargs.get("task") or "check my medications for today").strip()
        use_openai = bool(kwargs.get("use_openai", True))
        logger.info("Tool call: check_medications task=%s use_openai=%s", task, use_openai)

        result = run_medication_check(
            deps=deps,
            command=task,
            use_openai=use_openai,
        )

        first = result["first_pass"]
        second = result["verification_pass"]
        return {
            "status": "ok",
            "task": task,
            "image_path": result["image_path"],
            "summary": first["summary"],
            "verification_summary": second["summary"],
            "total_pills_visible": first["total_pills_visible"],
            "pills_by_color": first["pills_by_color"],
            "bottle_or_container_count": first["bottle_or_container_count"],
            "uncertainty": second["uncertainty"],
            "verification_notes": second["verification_notes"],
            "spoken_summary": first["summary"],
            "message_to_user_hint": (
                "Tell the user the total visible pill count first, then the color "
                "breakdown if available. If the image was unclear, say that plainly "
                "and ask them to keep the pills spread out in the tray and try again."
            ),
        }

"""Simple reminder tool — hardcoded one-shot timer for the Friday demo."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from reachy_medication_app.medication.store import get, list_all
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Module-level registry of pending reminders so the LLM can ask about them
_pending_reminders: list[dict[str, Any]] = []
_reminder_tasks: set[asyncio.Task[Any]] = set()
_lock = asyncio.Lock()


async def _fire_reminder(drug_name: str, message: str, deps: ToolDependencies, fire_at: float) -> None:
    """Wait until fire_at, then publish a reminder for the realtime loop to speak."""
    delay = max(0.0, fire_at - time.time())
    logger.info("reminder scheduled for %s in %.1fs: %s", drug_name, delay, message)
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return

    logger.info("REMINDER FIRED: %s — %s", drug_name, message)
    async with _lock:
        _pending_reminders.append(
            {
                "id": f"{drug_name}:{fire_at}",
                "drug_name": drug_name,
                "message": message,
                "fired_at": time.time(),
                "fired_at_str": time.strftime("%H:%M:%S"),
                "delivered": False,
            }
        )


class SetReminder(Tool):
    """Schedule a one-shot reminder for a medication at some seconds in the future."""

    name = "set_reminder"
    description = (
        "Schedule a one-time reminder for a medication. Use this when the user "
        "asks 'remind me in N minutes/hours to take X' or 'remind me at 8pm to "
        "take my Lisinopril'. For the demo, pass `delay_seconds` instead of an "
        "absolute time. Will fire once and only once."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Which medication this reminder is for. If omitted, use the only saved medication if there is one.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "How many seconds from now to fire the reminder. Use small values (30-300) for demoing.",
            },
            "message": {
                "type": "string",
                "description": "Optional custom reminder text. If omitted, a default will be generated.",
            },
        },
        "required": ["delay_seconds"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        drug_name = str(kwargs.get("drug_name", "")).strip()
        delay = int(kwargs.get("delay_seconds") or 0)
        custom_msg = str(kwargs.get("message", "")).strip()

        if delay <= 0:
            return {"status": "error", "error": "delay_seconds must be a positive integer"}

        if not drug_name:
            meds = list_all()
            if len(meds) == 1:
                drug_name = meds[0].drug_name
            else:
                drug_name = "your medication"

        med = get(drug_name)
        if med:
            default_msg = f"It's time to take your {med.drug_name} {med.strength}. {med.instructions}"
        else:
            default_msg = f"It's time to take your {drug_name}."

        message = custom_msg or default_msg
        fire_at = time.time() + delay

        # Spawn the background task. We do NOT await it.
        task = asyncio.create_task(_fire_reminder(drug_name, message, deps, fire_at))
        _reminder_tasks.add(task)
        task.add_done_callback(_reminder_tasks.discard)

        return {
            "status": "ok",
            "drug_name": drug_name,
            "delay_seconds": delay,
            "fire_at_str": time.strftime("%H:%M:%S", time.localtime(fire_at)),
            "message": message,
            "spoken_summary": f"Okay, I'll remind you about {drug_name} at {time.strftime('%H:%M:%S', time.localtime(fire_at))}.",
            "message_to_user_hint": (
                f"Confirm to the user that you'll remind them in about "
                f"{delay} seconds. Keep it warm — like 'Got it, I'll let you know "
                f"when it's time.'"
            ),
        }


class CheckPendingReminders(Tool):
    """Return any reminders that have fired but not been delivered yet.

    The conversational layer can call this periodically (or when the user pauses)
    to surface a reminder naturally. For the demo, the user can also just ask
    'do I have any reminders pending?' and the LLM will call this.
    """

    name = "check_pending_reminders"
    description = (
        "Check for any medication reminders that have fired but you haven't told "
        "the user about yet. Call this if the user asks 'do I need to take any "
        "medications?' or 'any reminders?'. Returns a list; if non-empty, say each "
        "reminder naturally and gently."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        logger.info("Tool call: check_pending_reminders")
        async with _lock:
            pending = [r for r in _pending_reminders if not r["delivered"]]
            for r in pending:
                r["delivered"] = True
        return {
            "status": "ok",
            "count": len(pending),
            "reminders": pending,
            "spoken_summary": (
                "You have no reminders waiting."
                if not pending
                else " ".join(str(r.get("message", "")).strip() for r in pending if str(r.get("message", "")).strip())
            ),
        }


async def pop_pending_reminders() -> list[dict[str, Any]]:
    """Return undelivered reminders without consuming them."""
    async with _lock:
        return [r for r in _pending_reminders if not r["delivered"]]


async def mark_reminders_delivered(reminders: list[dict[str, Any]]) -> None:
    """Mark specific pending reminders as delivered after they were actually announced."""
    if not reminders:
        return
    reminder_ids = {
        str(r.get("id"))
        for r in reminders
        if isinstance(r, dict) and str(r.get("id", "")).strip()
    }
    if not reminder_ids:
        return
    async with _lock:
        for reminder in _pending_reminders:
            if str(reminder.get("id")) in reminder_ids:
                reminder["delivered"] = True

"""Tray-counting tool for broad pill counting requests."""

from __future__ import annotations

import logging
from typing import Any

import cv2

from reachy_medication_app.medication.pipeline import (
    capture_multiview_frames,
    tray_scan_views,
)
from reachy_medication_app.medication.gemini_vlm import call_gemini
from reachy_medication_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _spoken_count_summary(total: Any, pills_by_color: dict[str, Any], confidence: str, uncertainty: list[str]) -> str:
    if total in (None, ""):
        return "I couldn't count the pills clearly. Please spread them out in the tray and let me try again."

    try:
        total_count = int(total)
    except Exception:
        return "I couldn't count the pills clearly. Please spread them out in the tray and let me try again."

    parts = [f"I see {total_count} pill{'s' if total_count != 1 else ''} total"]
    color_parts = []
    for color, count in pills_by_color.items():
        try:
            count_int = int(count)
        except Exception:
            continue
        color_parts.append(f"{count_int} {color}")
    if color_parts:
        parts.append(f"The colors look like {', '.join(color_parts)}")

    sentence = ". ".join(parts).strip(".") + "."
    if confidence == "low":
        detail = uncertainty[0] if uncertainty else "I'm not fully sure from this view."
        return f"{sentence} I may be off a little though. {detail}"
    return sentence


COUNT_PILLS_PROMPT = """You are helping an older adult count pills in a tray.

You are given multiple views of the same tray. Use the clearest view or combine views when counting.

Return ONLY valid JSON with this exact shape:
{
  "total_pills_visible": <integer or null>,
  "pills_by_color": {"<color>": <integer>},
  "summary": "<one short sentence saying how many pills you see>",
  "confidence": "<one of: high, medium, low>",
  "uncertainty": ["<brief note if the count is unclear or pills overlap>"]
}

Rules:
- Count only pills you can actually see.
- If one view is unclear but another view is clearer, use the clearest view.
- If you can see pills at all, give your best count instead of returning null.
- If you cannot count confidently, still give your best cautious estimate but set confidence to low and explain why in uncertainty.
- Use lowercase color names.
"""

SINGLE_VIEW_COUNT_PILLS_PROMPT = """You are helping an older adult count pills in a tray.

You are given one tray image. Count the pills you can actually see in this view.

Return ONLY valid JSON with this exact shape:
{
  "total_pills_visible": <integer or null>,
  "pills_by_color": {"<color>": <integer>},
  "summary": "<one short sentence saying how many pills you see>",
  "confidence": "<one of: high, medium, low>",
  "uncertainty": ["<brief note if the count is unclear or pills overlap>"]
}

Rules:
- Give your best count from this image if pills are visible.
- Do not return null unless you genuinely cannot see pills at all.
- Use lowercase color names.
"""


class CountPills(Tool):
    """Look down at the tray and count visible pills."""

    name = "count_pills"
    description = (
        "Look down at the tray in front of Reachy, count how many pills are visible, "
        "and report the color breakdown if available. Use this when the user asks "
        "general counting questions like 'how many pills are here', 'count these pills', "
        "or 'what pills do you see' without asking about a specific saved medication."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Optional natural-language tray counting request.",
            },
            "use_openai": {
                "type": "boolean",
                "description": "Whether to use OpenAI image analysis if available.",
                "default": True,
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        task = str(kwargs.get("task") or "count the pills in the tray").strip()
        logger.info("Tool call: count_pills task=%s", task)

        try:
            frames = capture_multiview_frames(deps, tray_scan_views())
        except Exception as e:
            logger.exception("count_pills: capture failed")
            return {
                "status": "error",
                "error": f"capture_failed: {e}",
                "spoken_summary": (
                    "I couldn't see the tray clearly. Please keep the pills in view and let me try again."
                ),
                "message_to_user_hint": (
                    "Tell the user you could not see the tray clearly and ask them to "
                    "keep the pills in view and try again."
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
                "message_to_user_hint": (
                    "Tell the user something went wrong while preparing the tray images "
                    "and ask them to try again."
                ),
            }

        result = call_gemini(COUNT_PILLS_PROMPT, jpeg_images)
        if result["error"]:
            return {
                "status": "error",
                "error": result["error"],
                "spoken_summary": (
                    "I had trouble counting the pills. Please spread them out in the tray and let me try again."
                ),
                "message_to_user_hint": (
                    "Tell the user you had trouble counting the pills and ask them to "
                    "spread them out in the tray and try again."
                ),
            }

        parsed = result["parsed_json"]
        needs_single_view_retry = (
            not parsed
            or parsed.get("total_pills_visible") in (None, "")
            or str(parsed.get("confidence", "low")).lower() == "low"
        )

        if needs_single_view_retry and jpeg_images:
            logger.info("count_pills: Gemini multi-view result unclear, retrying Gemini on center tray view")
            retry_result = call_gemini(SINGLE_VIEW_COUNT_PILLS_PROMPT, [jpeg_images[0]])
            retry_parsed = retry_result.get("parsed_json") if isinstance(retry_result, dict) else None
            if retry_parsed and retry_parsed.get("total_pills_visible") not in (None, ""):
                retry_uncertainty = retry_parsed.get("uncertainty", [])
                return {
                    "status": "ok",
                    "task": task,
                    "summary": retry_parsed.get("summary", ""),
                    "total_pills_visible": retry_parsed.get("total_pills_visible"),
                    "pills_by_color": retry_parsed.get("pills_by_color", {}),
                    "confidence": retry_parsed.get("confidence", "medium"),
                    "uncertainty": retry_uncertainty,
                    "spoken_summary": _spoken_count_summary(
                        retry_parsed.get("total_pills_visible"),
                        retry_parsed.get("pills_by_color", {}),
                        retry_parsed.get("confidence", "medium"),
                        retry_uncertainty if isinstance(retry_uncertainty, list) else [],
                    ),
                    "message_to_user_hint": (
                        "Tell the user the total visible pill count first, then the color "
                        "breakdown if available. If the count is still unclear, say that plainly "
                        "and ask them to keep the pills separated in the tray and try again."
                    ),
                }

        if not parsed:
            return {
                "status": "error",
                "error": "could_not_parse",
                "raw_text": result["raw_text"],
                "spoken_summary": (
                    "I got an unclear counting result. Please keep the pills separated in the tray and let me try again."
                ),
                "message_to_user_hint": (
                    "Tell the user you got an unclear counting result and ask them to "
                    "keep the pills separated in the tray and try again."
                ),
            }

        return {
            "status": "ok",
            "task": task,
            "summary": parsed.get("summary", ""),
            "total_pills_visible": parsed.get("total_pills_visible"),
            "pills_by_color": parsed.get("pills_by_color", {}),
            "confidence": parsed.get("confidence", "low"),
            "uncertainty": parsed.get("uncertainty", []),
            "spoken_summary": _spoken_count_summary(
                parsed.get("total_pills_visible"),
                parsed.get("pills_by_color", {}),
                parsed.get("confidence", "low"),
                parsed.get("uncertainty", []),
            ),
            "message_to_user_hint": (
                "Tell the user the total visible pill count first, then the color "
                "breakdown if available. If the count is unclear, say that plainly "
                "and ask them to keep the pills separated in the tray and try again."
            ),
        }

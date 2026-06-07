"""Shared medication-check helpers for the Reachy conversation app."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from reachy_mini.utils import create_head_pose
from reachy_medication_app.dance_emotion_moves import GotoQueueMove


@dataclass
class MedicationAnalysis:
    source: str
    summary: str
    total_pills_visible: Optional[int]
    pills_by_color: dict[str, int]
    bottle_or_container_count: Optional[int]
    bottles_or_containers_present: Optional[bool]
    confidence: str
    uncertainty: list[str]
    verification_notes: list[str]
    raw_model_output: Optional[dict[str, Any]] = None


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[..., ::-1].copy()


def save_ppm(rgb_image: np.ndarray, path: Path) -> None:
    header = f"P6\n{rgb_image.shape[1]} {rgb_image.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(rgb_image.tobytes())


def convert_ppm_to_png(ppm_path: Path, png_path: Path) -> bool:
    sips = shutil.which("sips")
    if sips is None:
        return False
    result = subprocess.run(
        [sips, "-s", "format", "png", str(ppm_path), "--out", str(png_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and png_path.exists()


def capture_frame(deps: Any) -> np.ndarray:
    """Get a frame from the conversation app camera worker or SDK media."""
    frame = None

    if getattr(deps, "camera_worker", None) is not None:
        frame = deps.camera_worker.get_latest_frame()

    if frame is None and getattr(deps, "reachy_mini", None) is not None:
        frame = deps.reachy_mini.media.get_frame()

    if frame is None:
        raise RuntimeError("No camera frame available from camera_worker or reachy_mini.media")

    return frame.astype(np.uint8)


def capture_and_save_frame(deps: Any, output_dir: Path | None = None) -> tuple[np.ndarray, Path]:
    frame = capture_frame(deps)
    target_dir = output_dir or Path(tempfile.mkdtemp(prefix="reachy_med_check_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    ppm_path = target_dir / "captured_frame.ppm"
    save_ppm(bgr_to_rgb(frame), ppm_path)

    png_path = target_dir / "captured_frame.png"
    if convert_ppm_to_png(ppm_path, png_path):
        return frame, png_path

    return frame, ppm_path


def _move_to_scan_pose(
    deps: Any,
    *,
    pitch_deg: float,
    yaw_deg: float,
    duration: float = 1.2,
    settle_seconds: float = 0.8,
) -> None:
    """Move Reachy using the shared movement manager so scans don't fight other motions."""
    movement_manager = deps.movement_manager
    movement_manager.clear_move_queue()

    current_head_pose = deps.reachy_mini.get_current_head_pose()
    head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()
    current_body_yaw = head_joints[0]

    target_head_pose = create_head_pose(0, 0, 0, 0, pitch_deg, yaw_deg, degrees=True)
    goto_move = GotoQueueMove(
        target_head_pose=target_head_pose,
        start_head_pose=current_head_pose,
        target_antennas=(antenna_joints[0], antenna_joints[1]),
        start_antennas=(antenna_joints[0], antenna_joints[1]),
        target_body_yaw=current_body_yaw,
        start_body_yaw=current_body_yaw,
        duration=duration,
    )
    movement_manager.queue_move(goto_move)
    movement_manager.set_moving_state(duration)
    time.sleep(max(0.0, duration + settle_seconds))


def capture_multiview_frames(
    deps: Any,
    views: list[dict[str, float]],
    *,
    settle_seconds: float = 0.8,
) -> list[np.ndarray]:
    """Capture one frame from each requested scan pose."""
    frames: list[np.ndarray] = []
    for view in views:
        _move_to_scan_pose(
            deps,
            pitch_deg=float(view["pitch_deg"]),
            yaw_deg=float(view["yaw_deg"]),
            duration=float(view.get("duration", 1.0)),
            settle_seconds=float(view.get("settle_seconds", settle_seconds)),
        )
        frames.append(capture_frame(deps))
    return frames


def tray_scan_views() -> list[dict[str, float]]:
    """Return a small set of tray-focused viewpoints."""
    return [
        {"pitch_deg": 32.0, "yaw_deg": 0.0, "duration": 1.0, "settle_seconds": 0.7},
        {"pitch_deg": 32.0, "yaw_deg": 12.0, "duration": 0.8, "settle_seconds": 0.5},
        {"pitch_deg": 32.0, "yaw_deg": -12.0, "duration": 0.8, "settle_seconds": 0.5},
    ]


def bottle_scan_views() -> list[dict[str, float]]:
    """Return a small set of bottle-focused viewpoints."""
    return [
        {"pitch_deg": -6.0, "yaw_deg": 0.0, "duration": 1.0, "settle_seconds": 0.7},
        {"pitch_deg": -2.0, "yaw_deg": 10.0, "duration": 0.8, "settle_seconds": 0.5},
        {"pitch_deg": -2.0, "yaw_deg": -10.0, "duration": 0.8, "settle_seconds": 0.5},
    ]


def connected_component_areas(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            area = 0

            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            areas.append(area)

    return areas


def heuristic_analysis(image: np.ndarray) -> MedicationAnalysis:
    working = image.astype(np.int16)
    yy, _xx = np.indices(image.shape[:2])

    bright_mask = (
        (working[..., 0] > 180)
        & (working[..., 1] > 180)
        & (working[..., 2] > 180)
        & (yy > image.shape[0] // 2)
    )
    red_mask = (
        (working[..., 2] > 150)
        & (working[..., 2] - working[..., 1] > 70)
        & (working[..., 2] - working[..., 0] > 70)
        & (yy > image.shape[0] // 2)
    )
    green_mask = (
        (working[..., 1] > 140)
        & (working[..., 1] - working[..., 2] > 50)
        & (working[..., 1] - working[..., 0] > 50)
        & (yy > image.shape[0] // 2)
    )
    bottle_mask = (
        (working[..., 2] > 140)
        & (working[..., 2] - working[..., 1] > 40)
        & (working[..., 2] - working[..., 0] > 50)
        & (yy < image.shape[0] * 0.8)
    )

    white_pills = sum(1 for area in connected_component_areas(bright_mask) if 180 <= area <= 1000)
    red_pills = sum(1 for area in connected_component_areas(red_mask) if 180 <= area <= 1200)
    green_pills = sum(1 for area in connected_component_areas(green_mask) if 180 <= area <= 1200)
    bottle_count = sum(1 for area in connected_component_areas(bottle_mask) if area >= 3000)

    pills_by_color = {
        color: count
        for color, count in {
            "white": white_pills,
            "red": red_pills,
            "green": green_pills,
        }.items()
        if count > 0
    }
    total = sum(pills_by_color.values()) if pills_by_color else None

    uncertainty: list[str] = []
    if total is None:
        uncertainty.append("No clear pill-like regions were detected.")
    if bottle_count == 0:
        uncertainty.append("No clear bottle/container regions were detected.")

    summary_parts: list[str] = []
    if total is not None:
        summary_parts.append(f"I estimate {total} visible pills.")
    else:
        summary_parts.append("I could not confidently count the visible pills.")
    if pills_by_color:
        color_text = ", ".join(f"{count} {color}" for color, count in pills_by_color.items())
        summary_parts.append(f"Approximate color breakdown: {color_text}.")
    if bottle_count > 0:
        summary_parts.append(f"I estimate {bottle_count} bottle/container objects.")

    return MedicationAnalysis(
        source="heuristic",
        summary=" ".join(summary_parts),
        total_pills_visible=total,
        pills_by_color=pills_by_color,
        bottle_or_container_count=bottle_count if bottle_count > 0 else None,
        bottles_or_containers_present=(bottle_count > 0),
        confidence="low" if uncertainty else "medium",
        uncertainty=uncertainty,
        verification_notes=[],
        raw_model_output=None,
    )


def encode_image_file_as_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = payload.get("output", [])
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise RuntimeError("Could not extract text from OpenAI response payload")


def normalize_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None
    return None


def normalize_color_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, int] = {}
    for key, raw in value.items():
        normalized = normalize_int(raw)
        if normalized is not None and normalized >= 0:
            cleaned[str(key).lower()] = normalized
    return cleaned


def openai_responses_request(
    api_key: str,
    model: str,
    image_data_url: str,
    prompt_text: str,
    detail: str,
) -> dict[str, Any]:
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are analyzing a tabletop medication scene captured from a small robot camera. "
                                "You must return valid JSON only."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": image_data_url, "detail": detail},
                    ],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def openai_first_pass(
    image_path: Path,
    command: str,
    model: str = "gpt-4.1-mini",
    detail: str = "high",
) -> MedicationAnalysis:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    image_data_url = encode_image_file_as_data_url(image_path)
    prompt = (
        "Analyze this image in the context of the spoken command: "
        f'"{command}". '
        "Return JSON with these keys exactly: "
        "summary, total_pills_visible, pills_by_color, bottle_or_container_count, "
        "bottles_or_containers_present, confidence, uncertainty. "
        "Count only visible pills. Mention uncertainty if counts may be affected by overlap, blur, glare, or occlusion."
    )
    payload = openai_responses_request(api_key, model, image_data_url, prompt, detail)
    parsed = json.loads(extract_output_text(payload))

    return MedicationAnalysis(
        source=f"openai:{model}",
        summary=str(parsed.get("summary", "")).strip(),
        total_pills_visible=normalize_int(parsed.get("total_pills_visible")),
        pills_by_color=normalize_color_counts(parsed.get("pills_by_color")),
        bottle_or_container_count=normalize_int(parsed.get("bottle_or_container_count")),
        bottles_or_containers_present=parsed.get("bottles_or_containers_present")
        if isinstance(parsed.get("bottles_or_containers_present"), bool)
        else None,
        confidence=str(parsed.get("confidence", "unknown")).lower(),
        uncertainty=[
            str(item)
            for item in parsed.get("uncertainty", [])
            if isinstance(item, (str, int, float))
        ],
        verification_notes=[],
        raw_model_output=parsed,
    )


def openai_verification_pass(
    image_path: Path,
    command: str,
    first_pass: MedicationAnalysis,
    model: str = "gpt-4.1-mini",
    detail: str = "high",
) -> MedicationAnalysis:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    image_data_url = encode_image_file_as_data_url(image_path)
    prompt = (
        "Verify the first-pass medication count for this image. "
        f'The user command was "{command}". '
        f"First pass result JSON: {json.dumps(asdict(first_pass), ensure_ascii=True)}. "
        "Check again carefully. Recount the pills. Mention uncertainty if pills overlap or image quality is poor. "
        "Return JSON with these keys exactly: "
        "summary, total_pills_visible, pills_by_color, bottle_or_container_count, "
        "bottles_or_containers_present, confidence, uncertainty, verification_notes."
    )
    payload = openai_responses_request(api_key, model, image_data_url, prompt, detail)
    parsed = json.loads(extract_output_text(payload))

    return MedicationAnalysis(
        source=f"openai_verify:{model}",
        summary=str(parsed.get("summary", "")).strip(),
        total_pills_visible=normalize_int(parsed.get("total_pills_visible")),
        pills_by_color=normalize_color_counts(parsed.get("pills_by_color")),
        bottle_or_container_count=normalize_int(parsed.get("bottle_or_container_count")),
        bottles_or_containers_present=parsed.get("bottles_or_containers_present")
        if isinstance(parsed.get("bottles_or_containers_present"), bool)
        else None,
        confidence=str(parsed.get("confidence", "unknown")).lower(),
        uncertainty=[
            str(item)
            for item in parsed.get("uncertainty", [])
            if isinstance(item, (str, int, float))
        ],
        verification_notes=[
            str(item)
            for item in parsed.get("verification_notes", [])
            if isinstance(item, (str, int, float))
        ],
        raw_model_output=parsed,
    )


def run_medication_check(
    deps: Any,
    command: str,
    use_openai: bool = True,
    model: str = "gpt-4.1-mini",
    image_detail: str = "high",
    settle_seconds: float = 2.0,
) -> dict[str, Any]:
    """Move the head down, capture an image, analyze it, and verify once."""
    _move_to_scan_pose(deps, pitch_deg=32.0, yaw_deg=0.0, duration=1.0, settle_seconds=settle_seconds)

    _frame, image_path = capture_and_save_frame(deps)

    if use_openai and os.getenv("OPENAI_API_KEY"):
        first = openai_first_pass(image_path=image_path, command=command, model=model, detail=image_detail)
        second = openai_verification_pass(
            image_path=image_path,
            command=command,
            first_pass=first,
            model=model,
            detail=image_detail,
        )
    else:
        first = heuristic_analysis(_frame)
        second = heuristic_analysis(_frame)
        second.verification_notes = [
            "Verification pass repeated the heuristic count.",
            "Switch to OpenAI vision analysis once the API key is configured for higher-quality scene understanding.",
        ]

    return {
        "image_path": str(image_path),
        "first_pass": asdict(first),
        "verification_pass": asdict(second),
    }

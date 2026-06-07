#!/usr/bin/env python3
"""Reachy Mini medication check.

- connect to Reachy Mini (real robot or local mockup simulator)
- tilt the head downward toward a medication area
- capture a camera frame
- optionally save the frame
- analyze the frame with either a heuristic fallback or an OpenAI vision model
- run a second verification pass that recounts and highlights uncertainty

Recommended real-robot usage:

    ./bin/python reachy_medication_check.py \
        --mode robot \
        --host reachy-mini.local \
        --analyzer openai \
        --command "check my medications for today"
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from dotenv import load_dotenv

try:
    from reachy_mini import ReachyMini
except Exception:  # pragma: no cover - optional import for mock-only usage
    ReachyMini = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "medication_check_outputs"
DEFAULT_COMMAND = "check my medications for today"


@dataclass
class CaptureResult:
    source: str
    note: str
    bgr_image: np.ndarray
    saved_ppm_path: Optional[str] = None
    saved_png_path: Optional[str] = None


def capture_result_to_report_dict(capture: CaptureResult) -> dict[str, Any]:
    return {
        "source": capture.source,
        "note": capture.note,
        "saved_ppm_path": capture.saved_ppm_path,
        "saved_png_path": capture.saved_png_path,
        "image_shape": list(capture.bgr_image.shape),
    }


@dataclass
class AnalysisResult:
    source: str
    summary: str
    total_pills_visible: Optional[int]
    pills_by_color: dict[str, int]
    bottle_or_container_count: Optional[int]
    bottles_or_containers_present: Optional[bool]
    confidence: str
    uncertainty: list[str]
    verification_notes: list[str] = field(default_factory=list)
    raw_model_output: Optional[dict[str, Any]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reachy Mini medication check demo")
    parser.add_argument(
        "--mode",
        choices=["robot", "mockup_sim", "mock"],
        default="mock",
        help="robot=connect to a physical Reachy Mini over the network; mockup_sim=local lightweight simulator; mock=offline fallback.",
    )
    parser.add_argument(
        "--host",
        default="reachy-mini.local",
        help="Robot hostname or IP. Used in --mode robot.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Reachy daemon port.",
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_COMMAND,
        help="High-level spoken-style command for the demo.",
    )
    parser.add_argument(
        "--analyzer",
        choices=["openai", "heuristic", "mock_vlm"],
        default="heuristic",
        help="Vision analysis backend.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1-mini",
        help="OpenAI model used when --analyzer openai.",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="high",
        help="Vision detail level for the OpenAI request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where captured images and reports should be written.",
    )
    parser.add_argument(
        "--head-target-x",
        type=float,
        default=0.32,
        help="Forward target in meters for look_at_world.",
    )
    parser.add_argument(
        "--head-target-y",
        type=float,
        default=0.0,
        help="Lateral target in meters for look_at_world.",
    )
    parser.add_argument(
        "--head-target-z",
        type=float,
        default=-0.16,
        help="Vertical target in meters for look_at_world. Negative values tilt downward.",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=1.5,
        help="Head movement duration in seconds.",
    )
    parser.add_argument(
        "--camera-warmup-seconds",
        type=float,
        default=2.0,
        help="How long to wait after movement before reading the camera.",
    )
    parser.add_argument(
        "--capture-attempts",
        type=int,
        default=6,
        help="Number of camera capture attempts before giving up.",
    )
    parser.add_argument(
        "--capture-interval-seconds",
        type=float,
        default=0.6,
        help="Pause between camera capture attempts.",
    )
    parser.add_argument(
        "--daemon-wait-seconds",
        type=float,
        default=4.0,
        help="How long to wait after spawning the local mockup simulator.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Use a synthetic scene if the real camera frame is unavailable.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def encode_image_file_as_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def draw_filled_circle(
    image: np.ndarray, center_x: int, center_y: int, radius: int, color: tuple[int, int, int]
) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    image[mask] = color


def draw_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    image[y0:y1, x0:x1] = color


def synthetic_medication_scene() -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = np.array([115, 83, 56], dtype=np.uint8)
    draw_rect(image, 0, 0, 640, 100, (175, 190, 205))

    for x0, y0, x1, y1 in ((80, 130, 165, 325), (235, 145, 315, 330)):
        draw_rect(image, x0, y0, x1, y1, (155, 70, 210))
        draw_rect(image, x0 + 10, y0 - 20, x1 - 10, y0, (235, 235, 235))

    for x, y, color in (
        (390, 320, (30, 30, 220)),
        (432, 312, (30, 30, 220)),
        (478, 338, (60, 185, 60)),
        (520, 308, (225, 225, 225)),
        (430, 360, (225, 225, 225)),
        (515, 362, (225, 225, 225)),
    ):
        draw_filled_circle(image, x, y, 12, color)
    return image


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


def heuristic_analysis(image: np.ndarray) -> AnalysisResult:
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
        summary_parts.append("I could not confidently count the pills.")
    if pills_by_color:
        color_text = ", ".join(f"{count} {color}" for color, count in pills_by_color.items())
        summary_parts.append(f"Approximate color breakdown: {color_text}.")
    if bottle_count > 0:
        summary_parts.append(f"I estimate {bottle_count} bottle/container objects.")

    return AnalysisResult(
        source="heuristic",
        summary=" ".join(summary_parts),
        total_pills_visible=total,
        pills_by_color=pills_by_color,
        bottle_or_container_count=bottle_count if bottle_count > 0 else None,
        bottles_or_containers_present=(bottle_count > 0) if bottle_count is not None else None,
        confidence="low" if uncertainty else "medium",
        uncertainty=uncertainty,
    )


def mock_vlm_analysis(image: np.ndarray, command: str) -> AnalysisResult:
    base = heuristic_analysis(image)
    uncertainty = list(base.uncertainty)
    if not uncertainty:
        uncertainty.append("This is a mock VLM response, not a real API call.")

    return AnalysisResult(
        source="mock_vlm",
        summary=(
            f'For the command "{command}", my mock VLM pass estimates '
            f"{base.total_pills_visible if base.total_pills_visible is not None else 'an uncertain number of'} pills "
            f"and {base.bottle_or_container_count if base.bottle_or_container_count is not None else 'an uncertain number of'} containers."
        ),
        total_pills_visible=base.total_pills_visible,
        pills_by_color=base.pills_by_color,
        bottle_or_container_count=base.bottle_or_container_count,
        bottles_or_containers_present=base.bottles_or_containers_present,
        confidence="low",
        uncertainty=uncertainty,
        raw_model_output={"note": "mock output"},
    )


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
    raise RuntimeError(f"Could not extract text from OpenAI response payload: {payload}")


def normalize_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
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
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": detail,
                        },
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
    model: str,
    detail: str,
) -> AnalysisResult:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell or put it in a .env file."
        )

    image_data_url = encode_image_file_as_data_url(image_path)
    prompt = (
        "Analyze this image in the context of the spoken command: "
        f'"{command}". '
        "Return JSON with these keys exactly: "
        "summary, total_pills_visible, pills_by_color, bottle_or_container_count, "
        "bottles_or_containers_present, confidence, uncertainty. "
        "Use a short summary. Count pills only if they are actually visible. "
        "If counts are uncertain because of overlap, glare, blur, or occlusion, say so in uncertainty."
    )

    payload = openai_responses_request(api_key, model, image_data_url, prompt, detail)
    parsed = json.loads(extract_output_text(payload))

    return AnalysisResult(
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
        raw_model_output=parsed,
    )


def openai_verification_pass(
    image_path: Path,
    command: str,
    first_pass: AnalysisResult,
    model: str,
    detail: str,
) -> AnalysisResult:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell or put it in a .env file."
        )

    image_data_url = encode_image_file_as_data_url(image_path)
    prompt = (
        "Verify the first-pass medication count for this same image. "
        f'The user command was "{command}". '
        f"First pass result JSON: {json.dumps(asdict(first_pass), ensure_ascii=True)}. "
        "Check again carefully. Recount the pills. Mention uncertainty if pills overlap or image quality is poor. "
        "Return JSON with these keys exactly: "
        "summary, total_pills_visible, pills_by_color, bottle_or_container_count, "
        "bottles_or_containers_present, confidence, uncertainty, verification_notes."
    )

    payload = openai_responses_request(api_key, model, image_data_url, prompt, detail)
    parsed = json.loads(extract_output_text(payload))

    return AnalysisResult(
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


class BaseController:
    def connect(self) -> None:
        raise NotImplementedError

    def move_head_down(self, x: float, y: float, z: float, duration: float) -> str:
        raise NotImplementedError

    def capture_image(
        self,
        attempts: int,
        interval_seconds: float,
        allow_synthetic_fallback: bool,
    ) -> CaptureResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockController(BaseController):
    def connect(self) -> None:
        print("Mock mode: no robot connection required.")

    def move_head_down(self, x: float, y: float, z: float, duration: float) -> str:
        return (
            "Mock movement only. "
            f"Would look toward world target ({x:.2f}, {y:.2f}, {z:.2f}) over {duration:.1f}s."
        )

    def capture_image(
        self,
        attempts: int,
        interval_seconds: float,
        allow_synthetic_fallback: bool,
    ) -> CaptureResult:
        return CaptureResult(
            source="synthetic",
            note="Using a synthetic medication scene in mock mode.",
            bgr_image=synthetic_medication_scene(),
        )

    def close(self) -> None:
        return


class ReachyController(BaseController):
    def __init__(self, mode: str, host: str, port: int, daemon_wait_seconds: float) -> None:
        self.mode = mode
        self.host = host
        self.port = port
        self.daemon_wait_seconds = daemon_wait_seconds
        self.reachy: Optional[Any] = None
        self.daemon_process: Optional[subprocess.Popen[str]] = None

    def connect(self) -> None:
        if ReachyMini is None:
            raise RuntimeError("reachy_mini is not available in this environment.")

        if self.mode == "mockup_sim":
            daemon_cmd = [
                str(ROOT / "bin" / "reachy-mini-daemon"),
                "--localhost-only",
                "--headless",
                "--no-media",
                "--no-wake-up-on-start",
                "--no-goto-sleep-on-stop",
                "--dataset-update-interval",
                "0",
                "--log-level",
                "INFO",
                "--mockup-sim",
            ]
            print(f"Starting local mockup simulator: {' '.join(daemon_cmd)}")
            self.daemon_process = subprocess.Popen(
                daemon_cmd,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            time.sleep(self.daemon_wait_seconds)

        connection_mode = "network" if self.mode == "robot" else "localhost_only"
        target_host = self.host if self.mode == "robot" else "localhost"

        self.reachy = ReachyMini(
            host=target_host,
            port=self.port,
            connection_mode=connection_mode,
            spawn_daemon=False,
            use_sim=False,
            media_backend="default",
            timeout=8.0,
        )
        print(f"Connected to Reachy Mini daemon at {target_host}:{self.port}.")

    def move_head_down(self, x: float, y: float, z: float, duration: float) -> str:
        if self.reachy is None:
            raise RuntimeError("Reachy connection is not initialized.")
        self.reachy.look_at_world(x=x, y=y, z=z, duration=duration, perform_movement=True)
        pose = self.reachy.get_current_head_pose().round(3).tolist()
        return f"Head movement completed. Current head pose={pose}"

    def capture_image(
        self,
        attempts: int,
        interval_seconds: float,
        allow_synthetic_fallback: bool,
    ) -> CaptureResult:
        if self.reachy is None:
            raise RuntimeError("Reachy connection is not initialized.")

        if getattr(self.reachy.media, "camera", None) is None:
            if allow_synthetic_fallback:
                return CaptureResult(
                    source="synthetic_fallback",
                    note=(
                        "Reachy media camera is not initialized in this mode. "
                        "Falling back to a synthetic scene."
                    ),
                    bgr_image=synthetic_medication_scene(),
                )
            raise RuntimeError(
                "Reachy media camera is not initialized. On the real robot, make sure the camera/media service is available."
            )

        last_exception: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                frame = self.reachy.media.get_frame()
                if frame is not None:
                    return CaptureResult(
                        source="reachy_camera",
                        note=f"Captured a frame from Reachy on attempt {attempt}.",
                        bgr_image=frame.astype(np.uint8),
                    )
            except Exception as exc:
                last_exception = exc
            time.sleep(interval_seconds)

        if allow_synthetic_fallback:
            note = "No camera frame was available. Falling back to a synthetic scene."
            if last_exception is not None:
                note += f" Last camera error: {last_exception}"
            return CaptureResult(
                source="synthetic_fallback",
                note=note,
                bgr_image=synthetic_medication_scene(),
            )

        if last_exception is not None:
            raise RuntimeError(f"Camera capture failed after {attempts} attempts: {last_exception}")
        raise RuntimeError(f"Camera capture returned no frame after {attempts} attempts.")

    def close(self) -> None:
        if self.reachy is not None:
            try:
                self.reachy.media.close()
            except Exception:
                pass
            try:
                self.reachy.client.disconnect()
            except Exception:
                pass
            self.reachy = None

        if self.daemon_process is not None and self.daemon_process.poll() is None:
            self.daemon_process.terminate()
            try:
                self.daemon_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.daemon_process.kill()
            self.daemon_process = None


def build_controller(args: argparse.Namespace) -> BaseController:
    if args.mode == "mock":
        return MockController()
    return ReachyController(args.mode, args.host, args.port, args.daemon_wait_seconds)


def save_capture(output_dir: Path, capture: CaptureResult) -> CaptureResult:
    rgb = bgr_to_rgb(capture.bgr_image)
    ppm_path = output_dir / "captured_frame.ppm"
    png_path = output_dir / "captured_frame.png"
    save_ppm(rgb, ppm_path)

    saved_png = None
    if convert_ppm_to_png(ppm_path, png_path):
        saved_png = str(png_path)

    capture.saved_ppm_path = str(ppm_path)
    capture.saved_png_path = saved_png
    return capture


def run_analysis(args: argparse.Namespace, capture: CaptureResult) -> tuple[AnalysisResult, AnalysisResult]:
    if args.analyzer == "heuristic":
        first = heuristic_analysis(capture.bgr_image)
        second = heuristic_analysis(capture.bgr_image)
        second.verification_notes = [
            "Verification pass repeated the heuristic count.",
            "If a real robot image contains overlap or glare, use the OpenAI analyzer for a stronger second pass.",
        ]
        return first, second

    if args.analyzer == "mock_vlm":
        first = mock_vlm_analysis(capture.bgr_image, args.command)
        second = mock_vlm_analysis(capture.bgr_image, args.command)
        second.verification_notes = [
            "Mock verification repeated the same synthetic reasoning path."
        ]
        return first, second

    if capture.saved_png_path is None:
        raise RuntimeError(
            "OpenAI analysis requires a PNG/JPEG/WEBP/GIF image. PNG export failed, so the image cannot be sent."
        )

    image_path = Path(capture.saved_png_path)
    first = openai_first_pass(
        image_path=image_path,
        command=args.command,
        model=args.openai_model,
        detail=args.image_detail,
    )
    second = openai_verification_pass(
        image_path=image_path,
        command=args.command,
        first_pass=first,
        model=args.openai_model,
        detail=args.image_detail,
    )
    return first, second


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    controller = build_controller(args)

    try:
        print(f'Command: "{args.command}"')
        print(f"Mode: {args.mode}")
        controller.connect()

        print("Step 1: moving head downward toward the medication area...")
        move_message = controller.move_head_down(
            x=args.head_target_x,
            y=args.head_target_y,
            z=args.head_target_z,
            duration=args.move_duration,
        )
        print(move_message)

        if args.camera_warmup_seconds > 0:
            print(f"Waiting {args.camera_warmup_seconds:.1f}s for the camera view to settle...")
            time.sleep(args.camera_warmup_seconds)

        print("Step 2: capturing a frame...")
        capture = controller.capture_image(
            attempts=args.capture_attempts,
            interval_seconds=args.capture_interval_seconds,
            allow_synthetic_fallback=args.allow_synthetic_fallback or args.mode == "mock",
        )
        capture = save_capture(args.output_dir, capture)
        print(capture.note)

        print(f"Step 3: running {args.analyzer} analysis...")
        first_pass, verification_pass = run_analysis(args, capture)

        report = {
            "command": args.command,
            "mode": args.mode,
            "host": args.host if args.mode == "robot" else "localhost",
            "analyzer": args.analyzer,
            "openai_model": args.openai_model if args.analyzer == "openai" else None,
            "movement": {
                "message": move_message,
                "target_world_m": {
                    "x": args.head_target_x,
                    "y": args.head_target_y,
                    "z": args.head_target_z,
                },
            },
            "capture": capture_result_to_report_dict(capture),
            "first_pass": asdict(first_pass),
            "verification_pass": asdict(verification_pass),
        }

        report_path = args.output_dir / "medication_check_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== Medication Check Summary ===")
        print(f"Capture source: {capture.source}")
        print(f"First pass: {first_pass.summary}")
        print(f"Verification: {verification_pass.summary}")
        if verification_pass.verification_notes:
            print("Verification notes:")
            for note in verification_pass.verification_notes:
                print(f"- {note}")
        if verification_pass.uncertainty:
            print("Uncertainty:")
            for note in verification_pass.uncertainty:
                print(f"- {note}")

        if first_pass.total_pills_visible is not None:
            print(f"Estimated visible pill count: {first_pass.total_pills_visible}")
        if first_pass.pills_by_color:
            print(f"Pills by color: {json.dumps(first_pass.pills_by_color, sort_keys=True)}")
        if first_pass.bottle_or_container_count is not None:
            print(f"Estimated containers: {first_pass.bottle_or_container_count}")

        if capture.saved_png_path:
            print(f"Saved PNG: {capture.saved_png_path}")
        print(f"Saved PPM: {capture.saved_ppm_path}")
        print(f"Saved report: {report_path}")
        return 0

    except Exception as exc:
        print(f"Medication check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Simulation-first Reachy Mini medication-scene prototype.

This script is designed for the first week of prototyping:
connect -> move head downward -> capture image (or fallback scene) ->
analyze scene -> verify again.

It supports three practical modes:
- mock: no daemon, no robot, always uses a synthetic scene
- mockup_sim: connect to a local Reachy Mini daemon started in mockup simulation
- mujoco_sim: connect to a local Reachy Mini daemon started in MuJoCo simulation

The analysis layer intentionally stays lightweight. It includes:
- a heuristic image analyzer for simple pills/bottles/count changes
- a mock VLM-style analyzer that produces a structured natural-language summary
- a second verification pass that recounts and highlights uncertainty
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from reachy_mini import ReachyMini
except Exception:  # pragma: no cover - optional import for mock-only usage
    ReachyMini = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "prototype_outputs"


@dataclass
class SceneMetadata:
    label: str
    true_pill_count: int
    true_bottle_count: int
    notes: list[str] = field(default_factory=list)


@dataclass
class SceneAnalysis:
    source: str
    summary: str
    pill_count_estimate: int
    bottle_count_estimate: int
    change_summary: str
    uncertainty: list[str]
    evidence: list[str]


@dataclass
class VerificationReport:
    verdict: str
    recounted_pill_count: int
    recounted_bottle_count: int
    confidence: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reachy Mini medication-scene prototype"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "mockup_sim", "mujoco_sim", "connect_local"],
        default="mock",
        help="Execution mode. 'mock' works without any daemon or robot.",
    )
    parser.add_argument(
        "--analyzer",
        choices=["heuristic", "mock_vlm"],
        default="heuristic",
        help="Analysis strategy to run on the captured scene.",
    )
    parser.add_argument(
        "--primary-scene",
        choices=["baseline", "changed", "overlap"],
        default="baseline",
        help="Synthetic scene used when camera capture is unavailable.",
    )
    parser.add_argument(
        "--comparison-scene",
        choices=["baseline", "changed", "overlap", "none"],
        default="changed",
        help="Optional second scene for a before/after change check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved images and JSON reports.",
    )
    parser.add_argument(
        "--head-target-x",
        type=float,
        default=0.35,
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
        default=-0.18,
        help="Vertical target in meters for look_at_world. Negative tilts downward.",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=1.5,
        help="Head movement duration in seconds.",
    )
    parser.add_argument(
        "--daemon-wait-seconds",
        type=float,
        default=4.0,
        help="How long to wait after spawning a local daemon before connecting.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_ppm(rgb_image: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 image as a binary PPM without extra dependencies."""
    if rgb_image.dtype != np.uint8:
        raise ValueError("PPM writer expects uint8 image data.")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("PPM writer expects shape (H, W, 3).")

    header = f"P6\n{rgb_image.shape[1]} {rgb_image.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(rgb_image.tobytes())


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected image with shape (H, W, 3).")
    return image[..., ::-1].copy()


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


def generate_synthetic_scene(name: str) -> tuple[np.ndarray, SceneMetadata]:
    """Build a simple tabletop medication scene for mock/demo mode."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = np.array([102, 78, 55], dtype=np.uint8)  # tabletop
    draw_rect(image, 0, 0, 640, 90, (165, 185, 205))  # back wall

    bottle_positions = [(90, 140, 170, 320), (250, 130, 330, 310)]
    pill_positions = [(410, 320), (455, 300), (500, 335), (540, 305), (455, 360)]
    notes: list[str] = []

    if name == "changed":
        bottle_positions = [(90, 140, 170, 320)]
        pill_positions = [(410, 320), (455, 300), (500, 335)]
        notes.append("One bottle removed and fewer pills remain on the table.")
    elif name == "overlap":
        bottle_positions = [(90, 140, 170, 320), (250, 130, 330, 310)]
        pill_positions = [(430, 320), (445, 325), (460, 330), (490, 300), (505, 305)]
        notes.append("Several pills overlap, making counting ambiguous.")

    for x0, y0, x1, y1 in bottle_positions:
        draw_rect(image, x0, y0, x1, y1, (70, 120, 210))
        draw_rect(image, x0 + 10, y0 - 22, x1 - 10, y0, (220, 220, 225))
        draw_rect(image, x0 + 18, y0 + 30, x1 - 18, y0 + 52, (240, 240, 245))

    for pill_x, pill_y in pill_positions:
        draw_filled_circle(image, pill_x, pill_y, 11, (245, 245, 238))
        draw_filled_circle(image, pill_x, pill_y, 4, (228, 228, 225))

    metadata = SceneMetadata(
        label=name,
        true_pill_count=len(pill_positions),
        true_bottle_count=len(bottle_positions),
        notes=notes,
    )
    return image, metadata


def connected_component_areas(mask: np.ndarray) -> list[int]:
    """Return connected component sizes for a boolean mask using 4-connectivity."""
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


def estimate_counts_from_image(image: np.ndarray) -> dict[str, Any]:
    """Very simple pixel heuristic for pills and bottles.

    Assumptions:
    - pills are bright and relatively small
    - bottles are blue-dominant and relatively large
    """
    image = image.astype(np.int16)

    yy, _xx = np.indices(image.shape[:2])

    pill_mask = (
        (image[..., 0] > 210)
        & (image[..., 1] > 210)
        & (image[..., 2] > 210)
        # Ignore bright bottle labels/caps higher in the image.
        & (yy > image.shape[0] // 2)
    )
    bottle_mask = (
        (image[..., 2] > 150)
        & (image[..., 2] - image[..., 1] > 40)
        & (image[..., 2] - image[..., 0] > 80)
    )

    pill_components = connected_component_areas(pill_mask)
    bottle_components = connected_component_areas(bottle_mask)

    pill_count = sum(1 for area in pill_components if 180 <= area <= 900)
    bottle_count = sum(1 for area in bottle_components if area >= 2500)

    overlap_warning = any(area > 900 for area in pill_components)
    low_signal_warning = pill_count == 0 and bottle_count == 0

    return {
        "pill_count": pill_count,
        "bottle_count": bottle_count,
        "pill_component_areas": pill_components,
        "bottle_component_areas": bottle_components,
        "overlap_warning": overlap_warning,
        "low_signal_warning": low_signal_warning,
    }


def compare_scenes(primary: dict[str, Any], comparison: Optional[dict[str, Any]]) -> str:
    if comparison is None:
        return "No comparison scene was provided."

    pill_delta = primary["pill_count"] - comparison["pill_count"]
    bottle_delta = primary["bottle_count"] - comparison["bottle_count"]
    changes: list[str] = []

    if pill_delta == 0:
        changes.append("pill estimate unchanged")
    elif pill_delta > 0:
        changes.append(f"{pill_delta} more pills visible in the primary scene")
    else:
        changes.append(f"{abs(pill_delta)} fewer pills visible in the primary scene")

    if bottle_delta == 0:
        changes.append("bottle estimate unchanged")
    elif bottle_delta > 0:
        changes.append(f"{bottle_delta} more bottles visible in the primary scene")
    else:
        changes.append(
            f"{abs(bottle_delta)} fewer bottles visible in the primary scene"
        )

    return "; ".join(changes) + "."


def heuristic_analysis(
    primary_image: np.ndarray, comparison_image: Optional[np.ndarray]
) -> SceneAnalysis:
    primary = estimate_counts_from_image(primary_image)
    comparison = (
        estimate_counts_from_image(comparison_image)
        if comparison_image is not None
        else None
    )

    uncertainty: list[str] = []
    evidence = [
        f"pill component areas={primary['pill_component_areas']}",
        f"bottle component areas={primary['bottle_component_areas']}",
    ]

    if primary["overlap_warning"]:
        uncertainty.append("Some bright pill-like regions merged together.")
    if primary["low_signal_warning"]:
        uncertainty.append("The heuristic found very little pill/bottle signal.")

    summary = (
        f"Estimated {primary['pill_count']} pills and {primary['bottle_count']} bottles "
        "in the primary scene."
    )

    return SceneAnalysis(
        source="heuristic",
        summary=summary,
        pill_count_estimate=primary["pill_count"],
        bottle_count_estimate=primary["bottle_count"],
        change_summary=compare_scenes(primary, comparison),
        uncertainty=uncertainty,
        evidence=evidence,
    )


def mock_vlm_analysis(
    primary_image: np.ndarray, comparison_image: Optional[np.ndarray]
) -> SceneAnalysis:
    base = heuristic_analysis(primary_image, comparison_image)
    uncertainty = list(base.uncertainty)
    if not uncertainty:
        uncertainty.append("This is a mock VLM-style interpretation, not a real model call.")

    summary = textwrap.dedent(
        f"""
        Mock VLM impression:
        - I see approximately {base.pill_count_estimate} pill-like objects.
        - I see approximately {base.bottle_count_estimate} bottle/container objects.
        - Change check: {base.change_summary}
        """
    ).strip()

    return SceneAnalysis(
        source="mock_vlm",
        summary=summary,
        pill_count_estimate=base.pill_count_estimate,
        bottle_count_estimate=base.bottle_count_estimate,
        change_summary=base.change_summary,
        uncertainty=uncertainty,
        evidence=base.evidence,
    )


def verification_pass(image: np.ndarray, first_pass: SceneAnalysis) -> VerificationReport:
    recount = estimate_counts_from_image(image)
    notes: list[str] = [
        "Verification prompt: check again carefully.",
        "Verification prompt: recount the pills.",
        "Verification prompt: mention uncertainty if pills overlap or image quality is bad.",
    ]

    if recount["pill_count"] != first_pass.pill_count_estimate:
        notes.append(
            "Recount differs from the first pass, so the pill count should be treated as uncertain."
        )
    if recount["overlap_warning"]:
        notes.append("The recount found merged bright regions that may hide overlapping pills.")
    if recount["low_signal_warning"]:
        notes.append("The recount had weak visual evidence.")

    if recount["pill_count"] == first_pass.pill_count_estimate:
        verdict = "Second pass agrees with the initial count."
        confidence = "medium"
    else:
        verdict = "Second pass disagrees with the initial count."
        confidence = "low"

    if recount["overlap_warning"] or recount["low_signal_warning"]:
        confidence = "low"

    return VerificationReport(
        verdict=verdict,
        recounted_pill_count=recount["pill_count"],
        recounted_bottle_count=recount["bottle_count"],
        confidence=confidence,
        notes=notes,
    )


class BaseController:
    def connect(self) -> None:
        raise NotImplementedError

    def move_head_down(self, x: float, y: float, z: float, duration: float) -> str:
        raise NotImplementedError

    def capture_image(self, scene_name: str) -> tuple[np.ndarray, SceneMetadata, str]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockController(BaseController):
    def connect(self) -> None:
        print("Mock mode: no Reachy daemon required.")

    def move_head_down(self, x: float, y: float, z: float, duration: float) -> str:
        return (
            "Mock head movement executed. "
            f"Requested look_at_world target=({x:.2f}, {y:.2f}, {z:.2f}) for {duration:.1f}s."
        )

    def capture_image(self, scene_name: str) -> tuple[np.ndarray, SceneMetadata, str]:
        image, metadata = generate_synthetic_scene(scene_name)
        return image, metadata, "Synthetic image generated locally."

    def close(self) -> None:
        return


class ReachySDKController(BaseController):
    def __init__(self, mode: str, daemon_wait_seconds: float) -> None:
        self.mode = mode
        self.daemon_wait_seconds = daemon_wait_seconds
        self.reachy: Optional[Any] = None
        self.daemon_process: Optional[subprocess.Popen[str]] = None

    def connect(self) -> None:
        if ReachyMini is None:
            raise RuntimeError(
                "reachy_mini is not importable in this environment. Use --mode mock."
            )

        if self.mode in {"mockup_sim", "mujoco_sim"}:
            daemon_cmd = [str(ROOT / "bin" / "reachy-mini-daemon")]
            daemon_cmd.extend(
                [
                    "--localhost-only",
                    "--headless",
                    "--no-media",
                    "--no-wake-up-on-start",
                    "--no-goto-sleep-on-stop",
                    "--dataset-update-interval",
                    "0",
                    "--log-level",
                    "INFO",
                ]
            )
            if self.mode == "mockup_sim":
                daemon_cmd.append("--mockup-sim")
            else:
                daemon_cmd.append("--sim")

            print(f"Starting local daemon: {' '.join(daemon_cmd)}")
            self.daemon_process = subprocess.Popen(
                daemon_cmd,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            time.sleep(self.daemon_wait_seconds)

        self.reachy = ReachyMini(
            host="localhost",
            port=8000,
            connection_mode="localhost_only",
            spawn_daemon=False,
            use_sim=False,
            media_backend="default",
            timeout=5.0,
        )
        print("Connected to Reachy Mini daemon on localhost.")

    def move_head_down(self, x: float, y: float, z: float, duration: float) -> str:
        if self.reachy is None:
            raise RuntimeError("Reachy client is not connected.")

        self.reachy.look_at_world(x=x, y=y, z=z, duration=duration, perform_movement=True)
        pose = self.reachy.get_current_head_pose().round(3).tolist()
        return f"Reachy head movement completed. Current head pose={pose}"

    def capture_image(self, scene_name: str) -> tuple[np.ndarray, SceneMetadata, str]:
        if self.reachy is None:
            raise RuntimeError("Reachy client is not connected.")

        frame = None
        try:
            frame = self.reachy.media.get_frame()
        except Exception as exc:
            print(f"Camera capture failed; falling back to synthetic scene. Reason: {exc}")

        if frame is not None:
            metadata = SceneMetadata(
                label="camera_capture",
                true_pill_count=-1,
                true_bottle_count=-1,
                notes=["Captured from Reachy media pipeline."],
            )
            return frame.astype(np.uint8), metadata, "Captured frame from Reachy media pipeline."

        image, metadata = generate_synthetic_scene(scene_name)
        return image, metadata, "Synthetic fallback image used because no camera frame was available."

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


def build_controller(mode: str, daemon_wait_seconds: float) -> BaseController:
    if mode == "mock":
        return MockController()
    return ReachySDKController(mode=mode, daemon_wait_seconds=daemon_wait_seconds)


def run_analysis(
    analyzer_name: str,
    primary_image: np.ndarray,
    comparison_image: Optional[np.ndarray],
) -> SceneAnalysis:
    if analyzer_name == "heuristic":
        return heuristic_analysis(primary_image, comparison_image)
    if analyzer_name == "mock_vlm":
        return mock_vlm_analysis(primary_image, comparison_image)
    raise ValueError(f"Unsupported analyzer: {analyzer_name}")


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    controller = build_controller(args.mode, args.daemon_wait_seconds)
    try:
        print(f"Prototype mode: {args.mode}")
        controller.connect()

        print("Step 1: moving head downward toward medication area...")
        move_message = controller.move_head_down(
            x=args.head_target_x,
            y=args.head_target_y,
            z=args.head_target_z,
            duration=args.move_duration,
        )
        print(move_message)

        print("Step 2: capturing primary scene...")
        primary_image, primary_metadata, primary_capture_note = controller.capture_image(
            args.primary_scene
        )
        print(primary_capture_note)

        comparison_image = None
        comparison_metadata: Optional[SceneMetadata] = None
        if args.comparison_scene != "none":
            print("Step 3: capturing comparison scene...")
            (
                comparison_image,
                comparison_metadata,
                comparison_capture_note,
            ) = controller.capture_image(args.comparison_scene)
            print(comparison_capture_note)

        primary_path = args.output_dir / "primary_scene.ppm"
        save_ppm(bgr_to_rgb(primary_image), primary_path)

        comparison_path = None
        if comparison_image is not None:
            comparison_path = args.output_dir / "comparison_scene.ppm"
            save_ppm(bgr_to_rgb(comparison_image), comparison_path)

        print(f"Step 4: analyzing with {args.analyzer}...")
        analysis = run_analysis(args.analyzer, primary_image, comparison_image)

        print("Step 5: running verification pass...")
        verification = verification_pass(primary_image, analysis)

        report = {
            "mode": args.mode,
            "analyzer": args.analyzer,
            "move_target_world_m": {
                "x": args.head_target_x,
                "y": args.head_target_y,
                "z": args.head_target_z,
            },
            "movement": move_message,
            "primary_capture": {
                "path": str(primary_path),
                "metadata": asdict(primary_metadata),
            },
            "comparison_capture": (
                {
                    "path": str(comparison_path) if comparison_path else None,
                    "metadata": asdict(comparison_metadata)
                    if comparison_metadata is not None
                    else None,
                }
                if args.comparison_scene != "none"
                else None
            ),
            "analysis": asdict(analysis),
            "verification": asdict(verification),
            "next_steps": [
                "Replace the mock analyzer with a real VLM call once camera viewpoints are stable.",
                "Collect 10-20 representative robot-view images before deciding whether a detector like YOLO is necessary.",
                "Use disagreement between first-pass and verification-pass counts as a trigger for human review.",
            ],
        }

        report_path = args.output_dir / "prototype_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== Prototype Summary ===")
        print(analysis.summary)
        print(f"Change check: {analysis.change_summary}")
        print(f"Verification: {verification.verdict}")
        print(f"Verification confidence: {verification.confidence}")
        if analysis.uncertainty:
            print("Uncertainty:")
            for note in analysis.uncertainty:
                print(f"- {note}")

        print(f"\nSaved primary image: {primary_path}")
        if comparison_path is not None:
            print(f"Saved comparison image: {comparison_path}")
        print(f"Saved report: {report_path}")
        return 0

    except Exception as exc:
        print(f"Prototype failed: {exc}", file=sys.stderr)
        return 1
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application import RunSummary, run_tracking  # noqa: E402
from src.config import AppConfig  # noqa: E402
from src.scene_config import SceneConfig  # noqa: E402


@dataclass(slots=True, frozen=True)
class DemoCase:
    key: str
    title: str
    source: Path
    scene: Path
    clip_start: float
    clip_duration: float
    clip_name: str


DEMO_CASES = (
    DemoCase(
        key="people",
        title="People flow monitoring",
        source=PROJECT_ROOT
        / "dataset/01_people_count/tocada_F1C22_walkway_130s_170s.mp4",
        scene=PROJECT_ROOT / "scenes/demo/people_count.yaml",
        clip_start=7.0,
        clip_duration=17.0,
        clip_name="01_people_count_h264.mp4",
    ),
    DemoCase(
        key="congestion",
        title="Gate traffic congestion",
        source=PROJECT_ROOT
        / "dataset/02_gate_congestion/pexels_14552311_fixed_view_traffic_jam.mp4",
        scene=PROJECT_ROOT / "scenes/demo/gate_congestion.yaml",
        clip_start=0.0,
        clip_duration=12.0,
        clip_name="02_gate_congestion_h264.mp4",
    ),
    DemoCase(
        key="clutter",
        title="Public area obstruction",
        source=PROJECT_ROOT
        / "dataset/03_public_area_clutter/caviar_LeftBox_abandoned_object.mpg",
        scene=PROJECT_ROOT / "scenes/demo/public_area_clutter.yaml",
        clip_start=20.0,
        clip_duration=14.5,
        clip_name="03_public_area_clutter_h264.mp4",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the three leadership demonstration videos."
    )
    parser.add_argument(
        "--only",
        choices=tuple(case.key for case in DEMO_CASES),
        help="Generate only one scene; the default generates all scenes.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the live preview while generating the result video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit each scene to this many frames for a quick smoke test.",
    )
    return parser


def run_case(case: DemoCase, show: bool, max_frames: int | None) -> RunSummary:
    if not case.source.is_file():
        raise FileNotFoundError(f"Demo source is missing: {case.source}")

    config = AppConfig.from_yaml(PROJECT_ROOT / "config.demo.yaml")
    config.display.show_window = show
    config.output.directory = PROJECT_ROOT / "outputs/leadership-demo" / case.key
    if config.output.directory.is_dir():
        shutil.rmtree(config.output.directory)
    scene = SceneConfig.from_yaml(case.scene)
    logging.info("Generating %s", case.title)
    return run_tracking(
        config,
        str(case.source),
        max_frames=max_frames,
        scene=scene,
    )


def create_ppt_clip(case: DemoCase, source: Path) -> Path:
    output_directory = PROJECT_ROOT / "outputs/leadership-demo/ppt"
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / case.clip_name
    command = [
        _ffmpeg_executable(),
        "-y",
        "-ss",
        str(case.clip_start),
        "-i",
        str(source),
        "-t",
        str(case.clip_duration),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Cannot create H.264 PPT clip:\n{message}")
    return output


def _ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "H.264 export requires ffmpeg or the imageio-ffmpeg package"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    args = build_parser().parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be greater than zero")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cases = (
        tuple(case for case in DEMO_CASES if case.key == args.only)
        if args.only
        else DEMO_CASES
    )
    for case in cases:
        summary = run_case(case, args.show, args.max_frames)
        logging.info(
            "Completed %s: %s (%d frames)",
            case.title,
            summary.video_path,
            summary.total_frames,
        )
        if summary.video_path is not None and args.max_frames is None:
            clip_path = create_ppt_clip(case, summary.video_path)
            logging.info("PPT H.264 clip: %s", clip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

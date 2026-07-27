from __future__ import annotations

import argparse
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

from src.application import run_tracking  # noqa: E402
from src.config import AppConfig, ConfigurationError  # noqa: E402
from src.detector_tracker import DetectionError  # noqa: E402
from src.result_writer import OutputError  # noqa: E402
from src.scene_config import SceneConfig  # noqa: E402
from src.video_source import SourceError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO and ByteTrack multi-object tracking MVP"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video path, camera index (for example 0), or RTSP URL",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="YAML configuration file",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="Scene YAML with regions, counting lines, and alert rules",
    )
    parser.add_argument("--model", help="Override the model path from config")
    parser.add_argument("--device", help="Override device: auto, cpu, 0, cuda:0")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the OpenCV preview window",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable annotated video output",
    )
    parser.add_argument(
        "--save-jsonl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable JSONL tracking output",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames; useful for smoke tests",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.model:
        config.model.path = args.model
    if args.device:
        config.model.device = args.device
    if args.output_dir:
        config.output.directory = args.output_dir.resolve()
    if args.show is not None:
        config.display.show_window = args.show
    if args.save_video is not None:
        config.output.save_video = args.save_video
    if args.save_jsonl is not None:
        config.output.save_jsonl = args.save_jsonl
    config.validate()
    return config


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    try:
        config = apply_cli_overrides(AppConfig.from_yaml(args.config), args)
        scene = SceneConfig.from_yaml(args.scene) if args.scene else None
        summary = run_tracking(
            config,
            args.source,
            max_frames=args.max_frames,
            scene=scene,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130
    except (ConfigurationError, SourceError, DetectionError, OutputError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info(
        "Finished: frames=%d elapsed=%.2fs average_fps=%.2f device=%s",
        summary.total_frames,
        summary.elapsed_seconds,
        summary.average_fps,
        summary.device,
    )
    if summary.video_path:
        logging.info("Result video: %s", summary.video_path)
    if summary.jsonl_path:
        logging.info("Tracking data: %s", summary.jsonl_path)
    if summary.metrics_path:
        logging.info("Scene metrics: %s", summary.metrics_path)
    if summary.events_path:
        logging.info("Alert events: %s", summary.events_path)
    if summary.summary_path:
        logging.info("Analytics summary: %s", summary.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

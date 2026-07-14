from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a clean obstruction baseline from a video or camera"
    )
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=60)
    return parser


def capture_baseline(
    source: str,
    start_seconds: float,
    duration_seconds: float,
    max_samples: int,
) -> np.ndarray:
    normalized_source: str | int = int(source) if source.isdecimal() else source
    capture = cv2.VideoCapture(normalized_source)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0
        if start_seconds > 0 and not isinstance(normalized_source, int):
            capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
        frame_limit = max(1, int(round(duration_seconds * fps)))
        stride = max(1, frame_limit // max_samples)
        frames: list[np.ndarray] = []
        for index in range(frame_limit):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                frames.append(frame)
                if len(frames) >= max_samples:
                    break
        if not frames:
            raise RuntimeError(f"Cannot read baseline frames from source: {source}")
        return np.median(np.stack(frames), axis=0).astype(np.uint8)
    finally:
        capture.release()


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise RuntimeError("Baseline output must use .jpg, .jpeg, or .png")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode baseline image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def main() -> int:
    args = build_parser().parse_args()
    if args.start_seconds < 0 or args.duration_seconds <= 0 or args.max_samples <= 0:
        raise SystemExit(
            "start must be non-negative; duration and samples must be positive"
        )
    baseline = capture_baseline(
        args.source,
        args.start_seconds,
        args.duration_seconds,
        args.max_samples,
    )
    write_image(args.output, baseline)
    print(f"Baseline written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

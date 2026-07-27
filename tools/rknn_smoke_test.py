from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import ModelConfig
from src.inference.rknn_detector import RknnDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RK3588 RKNN single-frame smoke test")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rknn-smoke.jpg")
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--npu-core", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read a frame from: {args.source}")

    config = ModelConfig(
        backend="rknn",
        path=str(args.model.expanduser().resolve()),
        confidence=args.confidence,
        npu_core=args.npu_core,
    )
    detector = RknnDetector(config, inference_confidence=args.confidence)
    try:
        detections = detector.detect(frame)
    finally:
        detector.close()

    for item in detections:
        x1, y1, x2, y2 = (int(value) for value in item.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            f"{item.class_name} {item.confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError(f"Cannot write result image: {args.output}")
    print(f"Detections: {len(detections)}")
    print(f"Result image: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an Ultralytics YOLO model to FP16 RKNN for RK3588."
    )
    parser.add_argument("--model", type=Path, required=True, help="Input .pt model")
    parser.add_argument("--target", default="rk3588", help="Rockchip target name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt model, got: {model_path}")

    from ultralytics import YOLO, __version__ as ultralytics_version

    print(f"Ultralytics: {ultralytics_version}")
    print(f"Input model: {model_path}")
    print(f"RKNN target: {args.target}")
    print("Quantization: disabled (FP16 on RK3588)")

    output = YOLO(str(model_path)).export(
        format="rknn",
        name=args.target,
        batch=1,
    )
    print(f"Exported RKNN package: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

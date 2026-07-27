from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "rknn-export"
    / "yolo11n_fp16_rknn_model"
    / "yolo11n_fp16.rknn"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an RK3588 deployment bundle")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--version",
        default=(PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip(),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dist")
    return parser


def build_bundle(
    project_root: Path,
    model_path: Path,
    output_dir: Path,
    version: str,
) -> Path:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"RKNN model does not exist: {model_path}")
    if model_path.suffix.lower() != ".rknn":
        raise ValueError(f"Expected an .rknn model: {model_path}")
    if not version or any(character in version for character in "\\/"):
        raise ValueError("Version must be a non-empty directory-safe value")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"park-safety-rk3588-{version}"
    bundle_dir = output_dir / bundle_name
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    if archive_path.exists():
        archive_path.unlink()

    bundle_dir.mkdir()
    shutil.copy2(project_root / "main.py", bundle_dir / "main.py")
    shutil.copytree(
        project_root / "src",
        bundle_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(project_root / "scenes", bundle_dir / "scenes")
    shutil.copy2(project_root / "config.rk3588.yaml", bundle_dir / "config.yaml")
    shutil.copy2(
        project_root / "requirements" / "rk3588.txt",
        bundle_dir / "requirements.txt",
    )
    shutil.copy2(project_root / "VERSION", bundle_dir / "VERSION")

    models_dir = bundle_dir / "models"
    models_dir.mkdir()
    shutil.copy2(model_path, models_dir / "yolo11n_fp16.rknn")

    tools_dir = bundle_dir / "tools"
    tools_dir.mkdir()
    shutil.copy2(
        project_root / "tools" / "rknn_smoke_test.py",
        tools_dir / "rknn_smoke_test.py",
    )
    for name in ("install.sh", "start.sh", "stop.sh", "park-safety.service"):
        shutil.copy2(project_root / "deploy" / "rk3588" / name, bundle_dir / name)

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(
            bundle_dir,
            arcname=bundle_name,
            filter=_deployment_tar_filter,
        )
    return archive_path


def _deployment_tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    if info.name.endswith(".sh"):
        info.mode = 0o755
    return info


def main() -> int:
    args = build_parser().parse_args()
    archive = build_bundle(
        PROJECT_ROOT,
        args.model,
        args.output_dir.expanduser().resolve(),
        args.version.strip(),
    )
    print(f"RK3588 bundle: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

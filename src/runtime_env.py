from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_ultralytics_directory() -> Path:
    """Return a writable Ultralytics config root and export it."""
    project_directory = PROJECT_ROOT / ".ultralytics"
    requested = Path(os.environ.get("YOLO_CONFIG_DIR", project_directory)).expanduser()
    try:
        requested.mkdir(parents=True, exist_ok=True)
        config_directory = requested
    except OSError:
        project_directory.mkdir(parents=True, exist_ok=True)
        config_directory = project_directory
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    return config_directory

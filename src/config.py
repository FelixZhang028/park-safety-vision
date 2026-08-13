from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when the application configuration is invalid."""


@dataclass(slots=True)
class ModelConfig:
    backend: str = "ultralytics"
    path: str = "yolo11n.pt"
    confidence: float = 0.35
    iou: float = 0.50
    image_size: int = 640
    device: str = "auto"
    npu_core: str = "auto"
    classes: tuple[int, ...] = (0, 2, 3, 5, 7)


@dataclass(slots=True)
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    persist: bool = True
    track_low_threshold: float = 0.10
    match_iou_threshold: float = 0.30
    track_buffer_frames: int = 30


@dataclass(slots=True)
class DisplayConfig:
    show_window: bool = True
    window_name: str = "Safety Vision Tracking"
    line_width: int = 2
    presentation_mode: bool = False
    show_fps: bool = True
    show_track_labels: bool = True
    output_width: int = 0
    output_height: int = 0


@dataclass(slots=True)
class OutputConfig:
    directory: Path = Path("outputs")
    save_video: bool = True
    save_jsonl: bool = True
    video_name: str = "tracked.mp4"
    jsonl_name: str = "tracks.jsonl"
    metrics_name: str = "metrics.jsonl"
    events_name: str = "events.jsonl"
    summary_name: str = "summary.json"
    event_images_directory: str = "events"


@dataclass(slots=True)
class RuntimeConfig:
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 1.0


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(
                f"Configuration file does not exist: {config_path}"
            )

        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Cannot read configuration: {exc}") from exc

        if not isinstance(raw, Mapping):
            raise ConfigurationError("The configuration root must be a mapping")

        _reject_unknown(
            raw, {"model", "tracking", "display", "output", "runtime"}, "root"
        )
        model_raw = _section(raw, "model")
        tracking_raw = _section(raw, "tracking")
        display_raw = _section(raw, "display")
        output_raw = _section(raw, "output")
        runtime_raw = _section(raw, "runtime")

        _reject_unknown(
            model_raw,
            {
                "backend",
                "path",
                "confidence",
                "iou",
                "image_size",
                "device",
                "npu_core",
                "classes",
            },
            "model",
        )
        _reject_unknown(
            tracking_raw,
            {
                "tracker",
                "persist",
                "track_low_threshold",
                "match_iou_threshold",
                "track_buffer_frames",
            },
            "tracking",
        )
        _reject_unknown(
            display_raw,
            {
                "show_window",
                "window_name",
                "line_width",
                "presentation_mode",
                "show_fps",
                "show_track_labels",
                "output_width",
                "output_height",
            },
            "display",
        )
        _reject_unknown(
            output_raw,
            {
                "directory",
                "save_video",
                "save_jsonl",
                "video_name",
                "jsonl_name",
                "metrics_name",
                "events_name",
                "summary_name",
                "event_images_directory",
            },
            "output",
        )
        _reject_unknown(
            runtime_raw,
            {"reconnect_attempts", "reconnect_delay_seconds"},
            "runtime",
        )

        base_dir = config_path.parent
        model_path = str(model_raw.get("path", "yolo11n.pt"))
        tracker_path = str(tracking_raw.get("tracker", "bytetrack.yaml"))

        config = cls(
            model=ModelConfig(
                backend=str(model_raw.get("backend", "ultralytics")),
                path=_resolve_named_or_relative(model_path, base_dir),
                confidence=float(model_raw.get("confidence", 0.35)),
                iou=float(model_raw.get("iou", 0.50)),
                image_size=int(model_raw.get("image_size", 640)),
                device=str(model_raw.get("device", "auto")),
                npu_core=str(model_raw.get("npu_core", "auto")),
                classes=tuple(
                    int(value) for value in model_raw.get("classes", [0, 2, 3, 5, 7])
                ),
            ),
            tracking=TrackingConfig(
                tracker=_resolve_named_or_relative(tracker_path, base_dir),
                persist=bool(tracking_raw.get("persist", True)),
                track_low_threshold=float(
                    tracking_raw.get("track_low_threshold", 0.10)
                ),
                match_iou_threshold=float(
                    tracking_raw.get("match_iou_threshold", 0.30)
                ),
                track_buffer_frames=int(
                    tracking_raw.get("track_buffer_frames", 30)
                ),
            ),
            display=DisplayConfig(
                show_window=bool(display_raw.get("show_window", True)),
                window_name=str(
                    display_raw.get("window_name", "Safety Vision Tracking")
                ),
                line_width=int(display_raw.get("line_width", 2)),
                presentation_mode=bool(
                    display_raw.get("presentation_mode", False)
                ),
                show_fps=bool(display_raw.get("show_fps", True)),
                show_track_labels=bool(
                    display_raw.get("show_track_labels", True)
                ),
                output_width=int(display_raw.get("output_width", 0)),
                output_height=int(display_raw.get("output_height", 0)),
            ),
            output=OutputConfig(
                directory=_resolve_directory(
                    output_raw.get("directory", "outputs"), base_dir
                ),
                save_video=bool(output_raw.get("save_video", True)),
                save_jsonl=bool(output_raw.get("save_jsonl", True)),
                video_name=str(output_raw.get("video_name", "tracked.mp4")),
                jsonl_name=str(output_raw.get("jsonl_name", "tracks.jsonl")),
                metrics_name=str(output_raw.get("metrics_name", "metrics.jsonl")),
                events_name=str(output_raw.get("events_name", "events.jsonl")),
                summary_name=str(output_raw.get("summary_name", "summary.json")),
                event_images_directory=str(
                    output_raw.get("event_images_directory", "events")
                ),
            ),
            runtime=RuntimeConfig(
                reconnect_attempts=int(runtime_raw.get("reconnect_attempts", 3)),
                reconnect_delay_seconds=float(
                    runtime_raw.get("reconnect_delay_seconds", 1.0)
                ),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.model.backend = self.model.backend.strip().lower()
        if self.model.backend not in {"ultralytics", "rknn"}:
            raise ConfigurationError(
                "model.backend must be 'ultralytics' or 'rknn'"
            )
        if not 0.0 <= self.model.confidence <= 1.0:
            raise ConfigurationError("model.confidence must be between 0 and 1")
        if not 0.0 <= self.model.iou <= 1.0:
            raise ConfigurationError("model.iou must be between 0 and 1")
        if self.model.image_size <= 0:
            raise ConfigurationError("model.image_size must be greater than zero")
        if not self.model.classes:
            raise ConfigurationError("model.classes cannot be empty")
        if any(class_id < 0 for class_id in self.model.classes):
            raise ConfigurationError(
                "model.classes must contain non-negative class IDs"
            )
        if self.model.backend == "rknn" and self.model.npu_core.strip().lower() not in {
            "auto",
            "0",
            "1",
            "2",
            "all",
        }:
            raise ConfigurationError(
                "model.npu_core must be auto, 0, 1, 2, or all"
            )
        if not 0.0 <= self.tracking.track_low_threshold <= 1.0:
            raise ConfigurationError(
                "tracking.track_low_threshold must be between 0 and 1"
            )
        if self.tracking.track_low_threshold > self.model.confidence:
            raise ConfigurationError(
                "tracking.track_low_threshold cannot exceed model.confidence"
            )
        if not 0.0 <= self.tracking.match_iou_threshold <= 1.0:
            raise ConfigurationError(
                "tracking.match_iou_threshold must be between 0 and 1"
            )
        if self.tracking.track_buffer_frames < 0:
            raise ConfigurationError(
                "tracking.track_buffer_frames cannot be negative"
            )
        if self.display.line_width <= 0:
            raise ConfigurationError("display.line_width must be greater than zero")
        if self.display.output_width < 0:
            raise ConfigurationError("display.output_width cannot be negative")
        if self.display.output_height < 0:
            raise ConfigurationError("display.output_height cannot be negative")
        if self.runtime.reconnect_attempts < 0:
            raise ConfigurationError("runtime.reconnect_attempts cannot be negative")
        if self.runtime.reconnect_delay_seconds < 0:
            raise ConfigurationError(
                "runtime.reconnect_delay_seconds cannot be negative"
            )
        if self.output.save_video and not self.output.video_name.lower().endswith(
            ".mp4"
        ):
            raise ConfigurationError("output.video_name must use the .mp4 extension")
        if self.output.save_jsonl and not self.output.jsonl_name.lower().endswith(
            ".jsonl"
        ):
            raise ConfigurationError("output.jsonl_name must use the .jsonl extension")

        if not self.output.metrics_name.lower().endswith(".jsonl"):
            raise ConfigurationError(
                "output.metrics_name must use the .jsonl extension"
            )
        if not self.output.events_name.lower().endswith(".jsonl"):
            raise ConfigurationError("output.events_name must use the .jsonl extension")
        if not self.output.summary_name.lower().endswith(".json"):
            raise ConfigurationError("output.summary_name must use the .json extension")
        if not self.output.event_images_directory.strip():
            raise ConfigurationError("output.event_images_directory cannot be empty")


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration key(s) in {section}: {', '.join(unknown)}"
        )


def _resolve_named_or_relative(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    return value


def _resolve_directory(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()

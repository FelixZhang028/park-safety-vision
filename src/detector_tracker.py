from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch

from .config import ModelConfig, TrackingConfig
from .runtime_env import configure_ultralytics_directory
from .schemas import TrackResult


LOGGER = logging.getLogger(__name__)
configure_ultralytics_directory()


class DetectionError(RuntimeError):
    """Raised when model loading or tracking fails."""


def _load_yolo(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectionError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        return YOLO(model_path)
    except Exception as exc:
        raise DetectionError(
            f"Cannot load model '{model_path}'. Check the path or network access: {exc}"
        ) from exc


class DetectorTracker:
    def __init__(
        self,
        model_config: ModelConfig,
        tracking_config: TrackingConfig,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_config = model_config
        self.tracking_config = tracking_config
        self.device = _select_device(model_config.device)
        factory = model_factory or _load_yolo
        self.model = factory(model_config.path)

    @property
    def device_description(self) -> str:
        if self.device == "cpu":
            return "CPU"
        index = int(self.device) if isinstance(self.device, int) else 0
        if torch.cuda.is_available():
            return f"CUDA:{index} ({torch.cuda.get_device_name(index)})"
        return str(self.device)

    def track(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> list[TrackResult]:
        try:
            results = self._run_model(frame)
        except RuntimeError as exc:
            if self.device != "cpu" and _is_cuda_failure(exc):
                LOGGER.warning("CUDA inference failed; retrying on CPU: %s", exc)
                self.device = "cpu"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    results = self._run_model(frame)
                except Exception as retry_exc:
                    raise DetectionError(
                        f"Tracking failed on CPU: {retry_exc}"
                    ) from retry_exc
            else:
                raise DetectionError(f"Tracking failed: {exc}") from exc
        except Exception as exc:
            raise DetectionError(f"Tracking failed: {exc}") from exc

        if not results:
            return []
        return parse_ultralytics_result(results[0], frame_id, timestamp)

    def _run_model(self, frame: np.ndarray):
        return self.model.track(
            source=frame,
            persist=self.tracking_config.persist,
            tracker=self.tracking_config.tracker,
            classes=list(self.model_config.classes),
            conf=self.model_config.confidence,
            iou=self.model_config.iou,
            imgsz=self.model_config.image_size,
            device=self.device,
            verbose=False,
        )


def parse_ultralytics_result(
    result: Any,
    frame_id: int,
    timestamp: float,
) -> list[TrackResult]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return []

    coordinates = _as_numpy(boxes.xyxy)
    class_ids = _as_numpy(boxes.cls).reshape(-1)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    track_ids = _as_numpy(boxes.id).reshape(-1)
    names = getattr(result, "names", {})

    tracks: list[TrackResult] = []
    for bbox, class_value, confidence, track_value in zip(
        coordinates, class_ids, confidences, track_ids, strict=True
    ):
        class_id = int(class_value)
        track_id = int(track_value)
        if track_id < 0:
            continue
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        else:
            class_name = str(names[class_id])
        tracks.append(
            TrackResult(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                bbox=tuple(float(value) for value in bbox[:4]),
                frame_id=frame_id,
                timestamp=timestamp,
            )
        )
    return tracks


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _select_device(requested: str) -> str | int:
    value = str(requested).strip().lower()
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if value.isdecimal():
        if torch.cuda.is_available():
            return int(value)
        LOGGER.warning("CUDA device %s requested but unavailable; using CPU", value)
        return "cpu"
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; using CPU")
            return "cpu"
        if ":" in value:
            return int(value.split(":", maxsplit=1)[1])
        return 0
    return requested


def _is_cuda_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("cuda", "cudnn", "cublas", "out of memory", "device-side")
    )

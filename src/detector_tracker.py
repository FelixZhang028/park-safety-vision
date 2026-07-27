from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .config import ModelConfig, TrackingConfig
from .inference.base import Detector
from .inference.factory import create_detector
from .inference.ultralytics_detector import parse_ultralytics_detections
from .schemas import TrackResult
from .tracking import ByteTrackAdapter


class DetectionError(RuntimeError):
    """Raised when model loading, inference, or tracking fails."""


class DetectorTracker:
    def __init__(
        self,
        model_config: ModelConfig,
        tracking_config: TrackingConfig,
        model_factory: Callable[[str], Any] | None = None,
        detector: Detector | None = None,
    ) -> None:
        self.model_config = model_config
        self.tracking_config = tracking_config
        try:
            self.detector = detector or create_detector(
                model_config,
                inference_confidence=min(
                    model_config.confidence,
                    tracking_config.track_low_threshold,
                ),
                model_factory=model_factory,
            )
        except Exception as exc:
            raise DetectionError(f"Cannot initialize detector: {exc}") from exc
        self.tracker = ByteTrackAdapter(
            tracking_config, high_threshold=model_config.confidence
        )

    @property
    def device_description(self) -> str:
        return self.detector.device_description

    def track(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> list[TrackResult]:
        try:
            detections = self.detector.detect(frame)
            return self.tracker.update(detections, frame_id, timestamp)
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"Detection or tracking failed: {exc}") from exc

    def close(self) -> None:
        self.detector.close()


def parse_ultralytics_result(
    result: Any,
    frame_id: int,
    timestamp: float,
) -> list[TrackResult]:
    """Compatibility helper for callers that already have Ultralytics track IDs."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return []
    detections = parse_ultralytics_detections(result)
    track_ids = _as_numpy(boxes.id).reshape(-1)
    return [
        TrackResult(
            track_id=int(track_id),
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        for detection, track_id in zip(detections, track_ids, strict=True)
        if int(track_id) >= 0
    ]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)

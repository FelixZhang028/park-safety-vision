from __future__ import annotations

from typing import Any, Callable

from ..config import ModelConfig
from .base import Detector


def create_detector(
    config: ModelConfig,
    inference_confidence: float | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> Detector:
    confidence = (
        config.confidence if inference_confidence is None else inference_confidence
    )
    if config.backend == "ultralytics":
        from .ultralytics_detector import UltralyticsDetector

        return UltralyticsDetector(
            config,
            inference_confidence=confidence,
            model_factory=model_factory,
        )
    if config.backend == "rknn":
        from .rknn_detector import RknnDetector

        return RknnDetector(config, inference_confidence=confidence)
    raise ValueError(f"Unsupported inference backend: {config.backend}")

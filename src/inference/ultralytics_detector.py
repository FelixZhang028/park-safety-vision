from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from ..config import ModelConfig
from ..runtime_env import configure_ultralytics_directory
from .base import Detector
from .schemas import DetectionResult


LOGGER = logging.getLogger(__name__)


class UltralyticsDetector(Detector):
    def __init__(
        self,
        config: ModelConfig,
        inference_confidence: float,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.inference_confidence = inference_confidence
        self._torch = _load_torch()
        self.device = _select_device(config.device, self._torch)
        factory = model_factory or _load_yolo
        self.model = factory(config.path)

    @property
    def device_description(self) -> str:
        if self.device == "cpu":
            return "CPU (Ultralytics)"
        index = int(self.device) if isinstance(self.device, int) else 0
        if self._torch.cuda.is_available():
            return f"CUDA:{index} ({self._torch.cuda.get_device_name(index)})"
        return f"{self.device} (Ultralytics)"

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        try:
            results = self._run_model(frame)
        except RuntimeError as exc:
            if self.device != "cpu" and _is_cuda_failure(exc):
                LOGGER.warning("CUDA inference failed; retrying on CPU: %s", exc)
                self.device = "cpu"
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                results = self._run_model(frame)
            else:
                raise
        if not results:
            return []
        return parse_ultralytics_detections(results[0])

    def _run_model(self, frame: np.ndarray):
        return self.model.predict(
            source=frame,
            classes=list(self.config.classes),
            conf=self.inference_confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.device,
            verbose=False,
        )


def parse_ultralytics_detections(result: Any) -> list[DetectionResult]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = _as_numpy(boxes.xyxy)
    class_ids = _as_numpy(boxes.cls).reshape(-1)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    names = getattr(result, "names", {})
    detections: list[DetectionResult] = []
    for bbox, class_value, confidence in zip(
        coordinates, class_ids, confidences, strict=True
    ):
        class_id = int(class_value)
        class_name = (
            str(names.get(class_id, class_id))
            if isinstance(names, dict)
            else str(names[class_id])
        )
        detections.append(
            DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                bbox=tuple(float(value) for value in bbox[:4]),
            )
        )
    return detections


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the ultralytics backend. "
            "Install requirements/desktop.txt."
        ) from exc
    return torch


def _load_yolo(model_path: str):
    configure_ultralytics_directory()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements/desktop.txt."
        ) from exc
    return YOLO(model_path)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _select_device(requested: str, torch_module: Any) -> str | int:
    value = str(requested).strip().lower()
    if value == "auto":
        return 0 if torch_module.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if value.isdecimal():
        if torch_module.cuda.is_available():
            return int(value)
        LOGGER.warning("CUDA device %s requested but unavailable; using CPU", value)
        return "cpu"
    if value.startswith("cuda"):
        if not torch_module.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; using CPU")
            return "cpu"
        return int(value.split(":", maxsplit=1)[1]) if ":" in value else 0
    return requested


def _is_cuda_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("cuda", "cudnn", "cublas", "out of memory", "device-side")
    )

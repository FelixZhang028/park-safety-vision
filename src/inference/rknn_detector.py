from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import ModelConfig
from .base import Detector
from .preprocessing import prepare_rknn_input
from .schemas import DetectionResult
from .yolo11_postprocess import decode_yolo11_output


class RknnDetector(Detector):
    def __init__(self, config: ModelConfig, inference_confidence: float) -> None:
        model_path = Path(config.path).expanduser()
        if not model_path.is_file():
            raise RuntimeError(f"RKNN model does not exist: {model_path}")
        self.config = config
        self.inference_confidence = inference_confidence
        rknn_lite_class = _load_rknn_lite()
        self._rknn = rknn_lite_class()
        ret = self._rknn.load_rknn(str(model_path))
        if ret != 0:
            self._rknn.release()
            raise RuntimeError(f"load_rknn failed with code {ret}: {model_path}")
        core_mask = _core_mask(rknn_lite_class, config.npu_core)
        ret = self._rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            self._rknn.release()
            raise RuntimeError(f"init_runtime failed with code {ret}")

    @property
    def device_description(self) -> str:
        return f"RK3588 NPU (core={self.config.npu_core})"

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        input_tensor, letterbox = prepare_rknn_input(
            frame, self.config.image_size
        )
        outputs = self._rknn.inference(inputs=[input_tensor])
        if not outputs:
            raise RuntimeError("RKNN inference returned no output")
        return decode_yolo11_output(
            outputs[0],
            letterbox=letterbox,
            confidence_threshold=self.inference_confidence,
            iou_threshold=self.config.iou,
            allowed_classes=self.config.classes,
        )

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


def _load_rknn_lite():
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit-lite2 is not installed. Install the RK3588 wheel first."
        ) from exc
    return RKNNLite


def _core_mask(rknn_lite_class: Any, requested: str) -> int:
    value = requested.strip().lower()
    names = {
        "auto": "NPU_CORE_AUTO",
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "all": "NPU_CORE_0_1_2",
    }
    return int(getattr(rknn_lite_class, names[value]))

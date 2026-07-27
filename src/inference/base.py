from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .schemas import DetectionResult


class Detector(ABC):
    @property
    @abstractmethod
    def device_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        raise NotImplementedError

    def close(self) -> None:
        pass

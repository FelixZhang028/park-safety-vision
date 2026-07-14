from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class FramePacket:
    frame: np.ndarray
    frame_id: int
    timestamp: float


@dataclass(slots=True, frozen=True)
class TrackResult:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    frame_id: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "bbox": [round(coordinate, 2) for coordinate in self.bbox],
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 6),
        }

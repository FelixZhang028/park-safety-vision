from __future__ import annotations

from dataclasses import dataclass


COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
}


@dataclass(slots=True, frozen=True)
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

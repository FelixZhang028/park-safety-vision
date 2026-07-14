from __future__ import annotations

from dataclasses import dataclass


NormalizedBox = tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)
class ObstructionCandidate:
    bbox: NormalizedBox
    area_ratio: float


@dataclass(slots=True, frozen=True)
class ObstructionDetection:
    candidates: tuple[ObstructionCandidate, ...] = ()
    changed_area_ratio: float = 0.0
    raw_change_ratio: float = 0.0
    scene_change_detected: bool = False

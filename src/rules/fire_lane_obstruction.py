from __future__ import annotations

from ..analytics.schemas import AlertEvent
from ..events.manager import EventManager
from ..obstruction.schemas import ObstructionDetection
from ..scene_config import FireLaneObstructionRuleConfig


class FireLaneObstructionRule:
    def __init__(
        self,
        config: FireLaneObstructionRuleConfig,
        scene_id: str,
    ) -> None:
        self.config = config
        self.scene_id = scene_id

    def update(
        self,
        detection: ObstructionDetection,
        manager: EventManager,
        timestamp: float,
        frame_id: int,
    ) -> tuple[bool, list[AlertEvent]]:
        if not self.config.enabled or not self.config.region:
            return False, []
        key = f"fire_lane_obstruction:{self.config.region}"
        boxes = [list(candidate.bbox) for candidate in detection.candidates]
        events = manager.evaluate(
            key=key,
            event_type="fire_lane_obstruction",
            condition=bool(detection.candidates)
            and not detection.scene_change_detected,
            severity="critical",
            scene_id=self.scene_id,
            region=self.config.region,
            timestamp=timestamp,
            frame_id=frame_id,
            hold_seconds=0.0,
            recovery_seconds=self.config.recovery_seconds,
            details={
                "changed_area_ratio": round(detection.changed_area_ratio, 6),
                "raw_change_ratio": round(detection.raw_change_ratio, 6),
                "obstruction_boxes": boxes,
                "scene_change_detected": detection.scene_change_detected,
            },
        )
        return manager.is_active(key), events

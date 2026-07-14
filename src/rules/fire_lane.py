from __future__ import annotations

from ..analytics.constants import VEHICLE_CLASSES
from ..analytics.schemas import AlertEvent, TrackMotion
from ..events.manager import EventManager
from ..scene_config import FireLaneRuleConfig, RegionConfig
from ..spatial.geometry import point_in_polygon


class FireLaneRule:
    PREFIX = "fire_lane:"

    def __init__(
        self,
        config: FireLaneRuleConfig,
        regions: dict[str, RegionConfig],
        scene_id: str,
    ) -> None:
        self.config = config
        self.regions = regions
        self.scene_id = scene_id

    def update(
        self,
        motions: list[TrackMotion],
        manager: EventManager,
        timestamp: float,
        frame_id: int,
    ) -> tuple[tuple[int, ...], list[AlertEvent]]:
        if not self.config.enabled or not self.config.region:
            return (), []
        region = self.regions[self.config.region]
        candidates = [
            motion
            for motion in motions
            if motion.track.class_name in VEHICLE_CLASSES
            and point_in_polygon(motion.point, region.polygon)
        ]
        seen_keys: set[str] = set()
        events: list[AlertEvent] = []
        for motion in candidates:
            key = f"{self.PREFIX}{self.config.region}:{motion.track.track_id}"
            seen_keys.add(key)
            events.extend(
                manager.evaluate(
                    key=key,
                    event_type="fire_lane_occupied",
                    condition=motion.speed_ratio <= self.config.max_speed_ratio,
                    severity="critical",
                    scene_id=self.scene_id,
                    region=self.config.region,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    hold_seconds=self.config.hold_seconds,
                    recovery_seconds=self.config.recovery_seconds,
                    track_ids=(motion.track.track_id,),
                    details={
                        "class_name": motion.track.class_name,
                        "speed_ratio": round(motion.speed_ratio, 6),
                    },
                )
            )
        events.extend(manager.mark_missing(self.PREFIX, seen_keys, timestamp, frame_id))
        return manager.active_track_ids(self.PREFIX), events

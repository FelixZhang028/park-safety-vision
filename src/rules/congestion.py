from __future__ import annotations

from dataclasses import dataclass

from ..analytics.constants import VEHICLE_CLASSES
from ..analytics.schemas import AlertEvent, TrackMotion
from ..events.manager import EventManager
from ..scene_config import CongestionRuleConfig, RegionConfig
from ..spatial.geometry import point_in_polygon


@dataclass(slots=True, frozen=True)
class CongestionMetrics:
    active: bool = False
    vehicle_count: int = 0
    mean_speed_ratio: float = 0.0
    low_speed_ratio: float = 0.0


class CongestionRule:
    def __init__(
        self,
        config: CongestionRuleConfig,
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
    ) -> tuple[CongestionMetrics, list[AlertEvent]]:
        if not self.config.enabled or not self.config.region:
            return CongestionMetrics(), []

        region = self.regions[self.config.region]
        vehicles = [
            motion
            for motion in motions
            if motion.track.class_name in VEHICLE_CLASSES
            and point_in_polygon(motion.point, region.polygon)
        ]
        speeds = [motion.speed_ratio for motion in vehicles]
        vehicle_count = len(vehicles)
        mean_speed = sum(speeds) / vehicle_count if vehicle_count else 0.0
        low_speed_count = sum(speed <= self.config.max_speed_ratio for speed in speeds)
        low_speed_ratio = low_speed_count / vehicle_count if vehicle_count else 0.0
        condition = (
            vehicle_count >= self.config.min_vehicles
            and low_speed_ratio >= self.config.min_low_speed_ratio
        )
        key = f"congestion:{self.config.region}"
        details = {
            "vehicle_count": vehicle_count,
            "mean_speed_ratio": round(mean_speed, 6),
            "low_speed_ratio": round(low_speed_ratio, 6),
        }
        events = manager.evaluate(
            key=key,
            event_type="traffic_congestion",
            condition=condition,
            severity="warning",
            scene_id=self.scene_id,
            region=self.config.region,
            timestamp=timestamp,
            frame_id=frame_id,
            hold_seconds=self.config.hold_seconds,
            recovery_seconds=self.config.recovery_seconds,
            track_ids=tuple(sorted(motion.track.track_id for motion in vehicles)),
            details=details,
        )
        return (
            CongestionMetrics(
                active=manager.is_active(key),
                vehicle_count=vehicle_count,
                mean_speed_ratio=mean_speed,
                low_speed_ratio=low_speed_ratio,
            ),
            events,
        )

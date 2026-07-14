from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import TrackResult
from ..spatial.geometry import Point


@dataclass(slots=True, frozen=True)
class TrackMotion:
    track: TrackResult
    point: Point
    previous_point: Point | None
    speed_ratio: float
    age_seconds: float


@dataclass(slots=True)
class AlertEvent:
    event_id: str
    event_type: str
    state: str
    severity: str
    scene_id: str
    region: str
    timestamp: float
    frame_id: int
    track_ids: tuple[int, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "state": self.state,
            "severity": self.severity,
            "scene_id": self.scene_id,
            "region": self.region,
            "timestamp": round(self.timestamp, 6),
            "frame_id": self.frame_id,
            "track_ids": list(self.track_ids),
            "details": self.details,
        }
        if self.snapshot_path:
            data["snapshot_path"] = self.snapshot_path
        return data


@dataclass(slots=True, frozen=True)
class AnalyticsSnapshot:
    scene_id: str
    frame_id: int
    timestamp: float
    current_people: int
    entries: int
    exits: int
    current_vehicles: int
    region_vehicle_counts: dict[str, int]
    congestion_active: bool
    congestion_vehicle_count: int
    mean_vehicle_speed_ratio: float
    low_speed_vehicle_ratio: float
    illegal_parking_track_ids: tuple[int, ...]
    fire_lane_track_ids: tuple[int, ...]
    active_event_types: tuple[str, ...]
    fire_lane_obstruction_active: bool = False
    fire_lane_obstruction_area_ratio: float = 0.0
    fire_lane_obstruction_boxes: tuple[tuple[float, float, float, float], ...] = ()
    fire_lane_scene_change_detected: bool = False
    current_visitors: int = 0
    visitor_loitering_pending: int = 0
    visitor_loitering_track_ids: tuple[int, ...] = ()
    visitor_loitering_regions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 6),
            "current_people": self.current_people,
            "entries": self.entries,
            "exits": self.exits,
            "current_vehicles": self.current_vehicles,
            "region_vehicle_counts": dict(self.region_vehicle_counts),
            "congestion_active": self.congestion_active,
            "congestion_vehicle_count": self.congestion_vehicle_count,
            "mean_vehicle_speed_ratio": round(self.mean_vehicle_speed_ratio, 6),
            "low_speed_vehicle_ratio": round(self.low_speed_vehicle_ratio, 6),
            "illegal_parking_track_ids": list(self.illegal_parking_track_ids),
            "fire_lane_track_ids": list(self.fire_lane_track_ids),
            "active_event_types": list(self.active_event_types),
            "fire_lane_obstruction_active": self.fire_lane_obstruction_active,
            "fire_lane_obstruction_area_ratio": round(
                self.fire_lane_obstruction_area_ratio, 6
            ),
            "fire_lane_obstruction_boxes": [
                list(box) for box in self.fire_lane_obstruction_boxes
            ],
            "fire_lane_scene_change_detected": (
                self.fire_lane_scene_change_detected
            ),
            "current_visitors": self.current_visitors,
            "visitor_loitering_pending": self.visitor_loitering_pending,
            "visitor_loitering_track_ids": list(
                self.visitor_loitering_track_ids
            ),
            "visitor_loitering_regions": list(self.visitor_loitering_regions),
        }


@dataclass(slots=True, frozen=True)
class AnalysisResult:
    snapshot: AnalyticsSnapshot
    events: tuple[AlertEvent, ...] = ()

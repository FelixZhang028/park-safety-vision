from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import numpy as np

from ..events.manager import EventManager
from ..identity.schemas import TrackIdentity
from ..obstruction import BackgroundObstructionDetector
from ..obstruction.schemas import ObstructionDetection
from ..rules import (
    CongestionRule,
    FireLaneObstructionRule,
    FireLaneRule,
    IllegalParkingRule,
    PersonCounter,
    VisitorLoiteringRule,
)
from ..scene_config import SceneConfig
from ..schemas import TrackResult
from ..spatial.geometry import point_in_polygon
from .constants import VEHICLE_CLASSES
from .schemas import AnalysisResult, AnalyticsSnapshot
from .track_history import TrackHistoryStore


class AnalyticsEngine:
    def __init__(self, scene: SceneConfig) -> None:
        self.scene = scene
        self.history = TrackHistoryStore(
            history_seconds=scene.history_seconds,
            stale_track_seconds=scene.stale_track_seconds,
        )
        self.events = EventManager()
        self.person_counter = PersonCounter(
            scene.person_count,
            scene.regions,
            scene.lines,
            scene.stale_track_seconds,
        )
        self.congestion = CongestionRule(
            scene.congestion, scene.regions, scene.scene_id
        )
        self.illegal_parking = IllegalParkingRule(
            scene.illegal_parking, scene.regions, scene.scene_id
        )
        self.fire_lane = FireLaneRule(scene.fire_lane, scene.regions, scene.scene_id)
        self.fire_lane_obstruction = FireLaneObstructionRule(
            scene.fire_lane_obstruction, scene.scene_id
        )
        self.obstruction_detector: BackgroundObstructionDetector | None = None
        if (
            scene.fire_lane_obstruction.enabled
            and scene.fire_lane_obstruction.region is not None
        ):
            self.obstruction_detector = BackgroundObstructionDetector(
                scene.fire_lane_obstruction,
                scene.regions[scene.fire_lane_obstruction.region],
            )
        self.visitor_loitering = VisitorLoiteringRule(
            scene.visitor_loitering, scene.regions, scene.scene_id
        )

    def update(
        self,
        tracks: list[TrackResult],
        timestamp: float,
        frame_id: int,
        frame_size: tuple[int, int],
        identities: Mapping[int, TrackIdentity] | None = None,
        observed_at: datetime | None = None,
        frame: np.ndarray | None = None,
    ) -> AnalysisResult:
        motions = self.history.update(tracks, timestamp, frame_size)
        identity_map = identities or {}
        person_stats = self.person_counter.update(motions, timestamp)
        congestion_metrics, congestion_events = self.congestion.update(
            motions, self.events, timestamp, frame_id
        )
        parking_ids, parking_events = self.illegal_parking.update(
            motions,
            self.events,
            timestamp,
            frame_id,
            congestion_metrics.active,
        )
        fire_lane_ids, fire_lane_events = self.fire_lane.update(
            motions, self.events, timestamp, frame_id
        )
        obstruction_detection = (
            self.obstruction_detector.update(frame, tracks, timestamp)
            if self.obstruction_detector is not None and frame is not None
            else ObstructionDetection()
        )
        obstruction_active, obstruction_events = self.fire_lane_obstruction.update(
            obstruction_detection, self.events, timestamp, frame_id
        )
        visitor_metrics, visitor_events = self.visitor_loitering.update(
            motions,
            identity_map,
            self.events,
            timestamp,
            frame_id,
            observed_at,
        )

        region_vehicle_counts = {
            name: sum(
                motion.track.class_name in VEHICLE_CLASSES
                and point_in_polygon(motion.point, region.polygon)
                for motion in motions
            )
            for name, region in self.scene.regions.items()
        }
        current_vehicles = sum(
            motion.track.class_name in VEHICLE_CLASSES for motion in motions
        )
        snapshot = AnalyticsSnapshot(
            scene_id=self.scene.scene_id,
            frame_id=frame_id,
            timestamp=timestamp,
            current_people=person_stats.current,
            entries=person_stats.entries,
            exits=person_stats.exits,
            current_vehicles=current_vehicles,
            region_vehicle_counts=region_vehicle_counts,
            congestion_active=congestion_metrics.active,
            congestion_vehicle_count=congestion_metrics.vehicle_count,
            mean_vehicle_speed_ratio=congestion_metrics.mean_speed_ratio,
            low_speed_vehicle_ratio=congestion_metrics.low_speed_ratio,
            illegal_parking_track_ids=parking_ids,
            fire_lane_track_ids=fire_lane_ids,
            active_event_types=self.events.active_event_types(),
            fire_lane_obstruction_active=obstruction_active,
            fire_lane_obstruction_area_ratio=(
                obstruction_detection.changed_area_ratio
            ),
            fire_lane_obstruction_boxes=tuple(
                candidate.bbox for candidate in obstruction_detection.candidates
            ),
            fire_lane_scene_change_detected=(
                obstruction_detection.scene_change_detected
            ),
            current_visitors=visitor_metrics.current_visitors,
            visitor_loitering_pending=visitor_metrics.pending_count,
            visitor_loitering_track_ids=visitor_metrics.active_track_ids,
            visitor_loitering_regions=visitor_metrics.active_regions,
        )
        return AnalysisResult(
            snapshot=snapshot,
            events=tuple(
                congestion_events
                + parking_events
                + fire_lane_events
                + obstruction_events
                + visitor_events
            ),
        )

    def reset(self) -> None:
        self.history.reset()
        self.events.reset()
        self.person_counter.reset()
        if self.obstruction_detector is not None:
            self.obstruction_detector.reset()
        self.visitor_loitering.reset()

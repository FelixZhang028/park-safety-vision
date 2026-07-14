from __future__ import annotations

import unittest

from src.analytics.engine import AnalyticsEngine
from src.scene_config import SceneConfig
from src.schemas import TrackResult


FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def make_track(
    track_id: int,
    class_name: str,
    point: tuple[float, float],
    timestamp: float,
    frame_id: int,
) -> TrackResult:
    x, y = point
    return TrackResult(
        track_id=track_id,
        class_id=0 if class_name == "person" else 2,
        class_name=class_name,
        confidence=0.9,
        bbox=(x * 100 - 5, y * 100 - 20, x * 100 + 5, y * 100),
        frame_id=frame_id,
        timestamp=timestamp,
    )


def scene_with_rules(**rules) -> SceneConfig:
    return SceneConfig.from_mapping(
        {
            "scene": {"id": "test_scene"},
            "regions": {
                "people": {"polygon": FULL_FRAME},
                "traffic": {"polygon": FULL_FRAME},
                "parking": {"polygon": FULL_FRAME},
                "fire": {"polygon": FULL_FRAME},
            },
            "lines": {
                "entry": {
                    "points": [[0.1, 0.5], [0.9, 0.5]],
                    "in_direction": "negative_to_positive",
                    "hysteresis": 0.01,
                    "cooldown_seconds": 0.1,
                }
            },
            "rules": rules,
            "analytics": {
                "history_seconds": 2,
                "stale_track_seconds": 1,
                "metrics_interval_seconds": 0.5,
            },
        }
    )


class AnalyticsEngineTests(unittest.TestCase):
    def test_person_crossing_is_counted_once(self) -> None:
        scene = scene_with_rules(
            person_count={
                "enabled": True,
                "region": "people",
                "lines": ["entry"],
            }
        )
        engine = AnalyticsEngine(scene)

        engine.update([make_track(1, "person", (0.5, 0.4), 0.0, 0)], 0.0, 0, (100, 100))
        engine.update(
            [make_track(1, "person", (0.5, 0.49), 0.2, 1)], 0.2, 1, (100, 100)
        )
        result = engine.update(
            [make_track(1, "person", (0.5, 0.6), 0.4, 2)],
            0.4,
            2,
            (100, 100),
        )
        repeated = engine.update(
            [make_track(1, "person", (0.5, 0.7), 0.6, 3)],
            0.6,
            3,
            (100, 100),
        )

        self.assertEqual(result.snapshot.entries, 1)
        self.assertEqual(repeated.snapshot.entries, 1)
        self.assertEqual(repeated.snapshot.exits, 0)
        self.assertEqual(repeated.snapshot.current_people, 1)

    def test_congestion_requires_duration_and_recovers(self) -> None:
        scene = scene_with_rules(
            congestion={
                "enabled": True,
                "region": "traffic",
                "min_vehicles": 2,
                "max_speed_ratio": 0.01,
                "min_low_speed_ratio": 1.0,
                "hold_seconds": 1.0,
                "recovery_seconds": 0.5,
            }
        )
        engine = AnalyticsEngine(scene)
        vehicles = [
            make_track(10, "car", (0.3, 0.7), 0.0, 0),
            make_track(11, "car", (0.6, 0.7), 0.0, 0),
        ]

        first = engine.update(vehicles, 0.0, 0, (100, 100))
        middle = engine.update(vehicles, 0.5, 1, (100, 100))
        active = engine.update(vehicles, 1.0, 2, (100, 100))
        recovering = engine.update([], 1.2, 3, (100, 100))
        ended = engine.update([], 1.7, 4, (100, 100))

        self.assertFalse(first.snapshot.congestion_active)
        self.assertFalse(middle.snapshot.congestion_active)
        self.assertTrue(active.snapshot.congestion_active)
        self.assertEqual(active.events[0].state, "started")
        self.assertTrue(recovering.snapshot.congestion_active)
        self.assertFalse(ended.snapshot.congestion_active)
        self.assertEqual(ended.events[0].state, "ended")

    def test_parking_and_fire_lane_events_are_track_specific(self) -> None:
        scene = scene_with_rules(
            illegal_parking={
                "enabled": True,
                "region": "parking",
                "max_speed_ratio": 0.01,
                "hold_seconds": 1.0,
                "recovery_seconds": 0.2,
                "suppress_when_congested": False,
            },
            fire_lane={
                "enabled": True,
                "region": "fire",
                "max_speed_ratio": 0.01,
                "hold_seconds": 0.5,
                "recovery_seconds": 0.2,
            },
        )
        engine = AnalyticsEngine(scene)

        def update(timestamp: float, frame_id: int):
            track = make_track(20, "truck", (0.5, 0.8), timestamp, frame_id)
            return engine.update([track], timestamp, frame_id, (100, 100))

        first = update(0.0, 0)
        fire_active = update(0.5, 1)
        both_active = update(1.0, 2)
        recovering = engine.update([], 1.1, 3, (100, 100))
        ended = engine.update([], 1.3, 4, (100, 100))

        self.assertFalse(first.snapshot.fire_lane_track_ids)
        self.assertEqual(fire_active.snapshot.fire_lane_track_ids, (20,))
        self.assertEqual(both_active.snapshot.illegal_parking_track_ids, (20,))
        self.assertTrue(recovering.snapshot.fire_lane_track_ids)
        self.assertFalse(ended.snapshot.fire_lane_track_ids)
        self.assertEqual(
            {event.event_type for event in ended.events},
            {"illegal_parking", "fire_lane_occupied"},
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime

from src.analytics.engine import AnalyticsEngine
from src.identity import TrackIdentity
from src.scene_config import SceneConfig
from src.schemas import TrackResult


FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def make_scene(
    *,
    period: str = "day",
    include_roles: list[str] | None = None,
) -> SceneConfig:
    return SceneConfig.from_mapping(
        {
            "scene": {"id": "visitor_test"},
            "regions": {"watch": {"polygon": FULL_FRAME}},
            "rules": {
                "visitor_loitering": {
                    "enabled": True,
                    "include_roles": include_roles or ["visitor"],
                    "period": period,
                    "day_start": "06:00",
                    "night_start": "18:00",
                    "zones": [
                        {
                            "region": "watch",
                            "day_hold_seconds": 2.0,
                            "night_hold_seconds": 1.0,
                            "absence_grace_seconds": 0.5,
                            "recovery_seconds": 0.4,
                        }
                    ],
                }
            },
            "analytics": {
                "history_seconds": 2,
                "stale_track_seconds": 1,
                "metrics_interval_seconds": 0.5,
            },
        }
    )


def make_track(track_id: int, timestamp: float, frame_id: int) -> TrackResult:
    return TrackResult(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(45.0, 30.0, 55.0, 60.0),
        frame_id=frame_id,
        timestamp=timestamp,
    )


def identity(
    track_id: int,
    role: str = "visitor",
    person_id: str | None = None,
) -> dict[int, TrackIdentity]:
    return {
        track_id: TrackIdentity(
            track_id=track_id,
            person_id=person_id,
            role=role,
            confidence=0.95,
        )
    }


class VisitorLoiteringTests(unittest.TestCase):
    def test_visitor_alert_starts_after_day_threshold(self) -> None:
        engine = AnalyticsEngine(make_scene())

        first = engine.update(
            [make_track(1, 0.0, 0)],
            0.0,
            0,
            (100, 100),
            identities=identity(1),
        )
        active = first
        for frame_id, timestamp in enumerate((0.5, 1.0, 1.5, 2.0), start=1):
            active = engine.update(
                [make_track(1, timestamp, frame_id)],
                timestamp,
                frame_id,
                (100, 100),
                identities=identity(1),
            )

        self.assertEqual(first.snapshot.visitor_loitering_pending, 1)
        self.assertFalse(first.events)
        self.assertEqual(active.snapshot.visitor_loitering_track_ids, (1,))
        self.assertEqual(active.events[0].event_type, "visitor_abnormal_stay")
        self.assertEqual(active.events[0].details["period"], "day")
        self.assertEqual(active.events[0].details["dwell_seconds"], 2.0)

    def test_employee_is_not_treated_as_visitor(self) -> None:
        engine = AnalyticsEngine(make_scene(include_roles=["visitor"]))

        result = engine.update(
            [make_track(2, 3.0, 0)],
            3.0,
            0,
            (100, 100),
            identities=identity(2, role="employee"),
        )

        self.assertEqual(result.snapshot.current_visitors, 0)
        self.assertEqual(result.snapshot.visitor_loitering_pending, 0)
        self.assertFalse(result.events)

    def test_short_absence_is_tolerated_and_long_absence_ends_event(self) -> None:
        engine = AnalyticsEngine(make_scene())

        engine.update(
            [make_track(3, 0.0, 0)],
            0.0,
            0,
            (100, 100),
            identities=identity(3),
        )
        brief_gap = engine.update([], 0.3, 1, (100, 100))
        engine.update(
            [make_track(3, 0.4, 2)],
            0.4,
            2,
            (100, 100),
            identities=identity(3),
        )
        active = brief_gap
        for frame_id, timestamp in enumerate((0.8, 1.2, 1.6, 2.0), start=3):
            active = engine.update(
                [make_track(3, timestamp, frame_id)],
                timestamp,
                frame_id,
                (100, 100),
                identities=identity(3),
            )
        engine.update([], 2.3, 7, (100, 100))
        recovering = engine.update([], 2.6, 8, (100, 100))
        ended = engine.update([], 3.0, 9, (100, 100))

        self.assertEqual(brief_gap.snapshot.visitor_loitering_pending, 1)
        self.assertEqual(active.snapshot.visitor_loitering_track_ids, (3,))
        self.assertEqual(recovering.snapshot.visitor_loitering_track_ids, (3,))
        self.assertFalse(ended.snapshot.visitor_loitering_track_ids)
        self.assertEqual(ended.events[0].state, "ended")

    def test_stable_person_identity_survives_track_id_change(self) -> None:
        engine = AnalyticsEngine(make_scene())

        engine.update(
            [make_track(10, 0.0, 0)],
            0.0,
            0,
            (100, 100),
            identities=identity(10, person_id="visitor-42"),
        )
        active = None
        for frame_id, timestamp in enumerate((0.4, 0.8, 1.2, 1.6, 2.0), start=1):
            active = engine.update(
                [make_track(11, timestamp, frame_id)],
                timestamp,
                frame_id,
                (100, 100),
                identities=identity(11, person_id="visitor-42"),
            )

        assert active is not None
        self.assertEqual(active.snapshot.visitor_loitering_track_ids, (11,))
        self.assertEqual(active.events[0].details["person_id"], "visitor-42")

    def test_auto_period_uses_night_threshold(self) -> None:
        engine = AnalyticsEngine(make_scene(period="auto"))
        night = datetime(2026, 1, 1, 22, 0)

        engine.update(
            [make_track(20, 0.0, 0)],
            0.0,
            0,
            (100, 100),
            identities=identity(20),
            observed_at=night,
        )
        engine.update(
            [make_track(20, 0.5, 1)],
            0.5,
            1,
            (100, 100),
            identities=identity(20),
            observed_at=night,
        )
        active = engine.update(
            [make_track(20, 1.0, 2)],
            1.0,
            2,
            (100, 100),
            identities=identity(20),
            observed_at=night,
        )

        self.assertEqual(active.events[0].severity, "critical")
        self.assertEqual(active.events[0].details["period"], "night")
        self.assertEqual(active.events[0].details["threshold_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()

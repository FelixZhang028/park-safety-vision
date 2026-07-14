from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.events.manager import EventManager
from src.obstruction.background_detector import BackgroundObstructionDetector
from src.obstruction.schemas import ObstructionCandidate, ObstructionDetection
from src.rules.fire_lane_obstruction import FireLaneObstructionRule
from src.scene_config import (
    FireLaneObstructionRuleConfig,
    RegionConfig,
)
from src.schemas import TrackResult


FULL_FRAME = RegionConfig(
    name="fire",
    polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
)


class BackgroundObstructionDetectorTests(unittest.TestCase):
    def test_stable_unknown_change_is_detected_and_person_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.jpg"
            cv2.imwrite(str(baseline_path), np.full((100, 100, 3), 30, np.uint8))
            config = FireLaneObstructionRuleConfig(
                enabled=True,
                region="fire",
                baseline_path=baseline_path,
                pixel_threshold=20,
                min_area_ratio=0.01,
                hold_seconds=1.0,
                recovery_seconds=0.5,
                exclusion_padding_ratio=0.0,
                max_global_change_ratio=0.8,
            )
            detector = BackgroundObstructionDetector(config, FULL_FRAME)
            frame = np.full((100, 100, 3), 30, np.uint8)
            frame[30:65, 30:65] = 220

            self.assertFalse(detector.update(frame, [], 0.0).candidates)
            self.assertFalse(detector.update(frame, [], 0.5).candidates)
            detected = detector.update(frame, [], 1.0)
            self.assertTrue(detected.candidates)
            self.assertGreater(detected.changed_area_ratio, 0.01)

            person = TrackResult(
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(25.0, 25.0, 70.0, 70.0),
                frame_id=3,
                timestamp=1.1,
            )
            self.assertFalse(detector.update(frame, [person], 1.1).candidates)

    def test_large_global_change_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.jpg"
            cv2.imwrite(str(baseline_path), np.zeros((80, 80, 3), np.uint8))
            config = FireLaneObstructionRuleConfig(
                enabled=True,
                region="fire",
                baseline_path=baseline_path,
                pixel_threshold=10,
                min_area_ratio=0.01,
                hold_seconds=0.0,
                max_global_change_ratio=0.5,
            )
            detector = BackgroundObstructionDetector(config, FULL_FRAME)

            result = detector.update(np.full((80, 80, 3), 255, np.uint8), [], 0.0)

            self.assertTrue(result.scene_change_detected)
            self.assertFalse(result.candidates)

            clear = detector.update(np.zeros((80, 80, 3), np.uint8), [], 1.0)
            self.assertFalse(clear.scene_change_detected)
            self.assertFalse(clear.candidates)


class FireLaneObstructionRuleTests(unittest.TestCase):
    def test_alert_starts_and_recovers(self) -> None:
        config = FireLaneObstructionRuleConfig(
            enabled=True,
            region="fire",
            baseline_path=Path("unused.jpg"),
            recovery_seconds=1.0,
        )
        rule = FireLaneObstructionRule(config, "scene")
        manager = EventManager()
        detection = ObstructionDetection(
            candidates=(ObstructionCandidate((0.1, 0.2, 0.3, 0.4), 0.02),),
            changed_area_ratio=0.02,
        )

        active, started = rule.update(detection, manager, 2.0, 20)
        recovering, middle = rule.update(ObstructionDetection(), manager, 2.5, 25)
        ended_active, ended = rule.update(ObstructionDetection(), manager, 3.5, 35)

        self.assertTrue(active)
        self.assertEqual(started[0].state, "started")
        self.assertTrue(recovering)
        self.assertFalse(middle)
        self.assertFalse(ended_active)
        self.assertEqual(ended[0].state, "ended")


if __name__ == "__main__":
    unittest.main()

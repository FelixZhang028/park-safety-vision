from __future__ import annotations

import unittest

from src.config import TrackingConfig
from src.inference.schemas import DetectionResult
from src.tracking import ByteTrackAdapter


def detection(confidence: float, x: float = 10.0) -> DetectionResult:
    return DetectionResult(
        class_id=2,
        class_name="car",
        confidence=confidence,
        bbox=(x, 10.0, x + 100.0, 80.0),
    )


class ByteTrackAdapterTests(unittest.TestCase):
    def test_track_id_survives_motion_and_low_confidence_frame(self) -> None:
        tracker = ByteTrackAdapter(
            TrackingConfig(
                track_low_threshold=0.10,
                match_iou_threshold=0.30,
                track_buffer_frames=2,
            ),
            high_threshold=0.35,
        )

        first = tracker.update([detection(0.90)], 1, 0.0)
        second = tracker.update([detection(0.20, x=14.0)], 2, 0.1)
        third = tracker.update([detection(0.88, x=18.0)], 3, 0.2)

        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)
        self.assertEqual(third[0].track_id, 1)

    def test_expired_track_gets_new_id(self) -> None:
        tracker = ByteTrackAdapter(
            TrackingConfig(track_buffer_frames=1),
            high_threshold=0.35,
        )
        first = tracker.update([detection(0.9)], 1, 0.0)
        tracker.update([], 2, 0.1)
        tracker.update([], 3, 0.2)
        current = tracker.update([detection(0.9)], 4, 0.3)

        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(current[0].track_id, 2)


if __name__ == "__main__":
    unittest.main()

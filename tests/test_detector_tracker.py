from __future__ import annotations

import unittest

import numpy as np

from src.config import ModelConfig, TrackingConfig
from src.detector_tracker import DetectorTracker, parse_ultralytics_result


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array(
            [[10.0, 20.0, 110.0, 220.0], [30.0, 40.0, 130.0, 240.0]],
            dtype=np.float32,
        )
        self.cls = np.array([0.0, 2.0], dtype=np.float32)
        self.conf = np.array([0.91, 0.82], dtype=np.float32)
        self.id = np.array([7.0, 9.0], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person", 2: "car"}


class FakeModel:
    def __init__(self) -> None:
        self.last_kwargs = None

    def track(self, **kwargs):
        self.last_kwargs = kwargs
        return [FakeResult()]


class DetectorTrackerTests(unittest.TestCase):
    def test_result_is_converted_to_required_schema(self) -> None:
        tracks = parse_ultralytics_result(FakeResult(), frame_id=12, timestamp=0.4)

        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].track_id, 7)
        self.assertEqual(tracks[0].class_name, "person")
        self.assertEqual(tracks[0].bbox, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(tracks[0].frame_id, 12)
        self.assertEqual(tracks[0].timestamp, 0.4)

    def test_ultralytics_builtin_tracking_arguments_are_used(self) -> None:
        fake_model = FakeModel()
        detector = DetectorTracker(
            ModelConfig(device="cpu"),
            TrackingConfig(tracker="bytetrack.yaml", persist=True),
            model_factory=lambda _: fake_model,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        tracks = detector.track(frame, frame_id=3, timestamp=0.1)

        self.assertEqual(len(tracks), 2)
        self.assertEqual(fake_model.last_kwargs["tracker"], "bytetrack.yaml")
        self.assertTrue(fake_model.last_kwargs["persist"])
        self.assertEqual(fake_model.last_kwargs["classes"], [0, 2, 3, 5, 7])
        self.assertEqual(fake_model.last_kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()

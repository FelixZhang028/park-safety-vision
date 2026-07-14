from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import OutputConfig
from src.result_writer import ResultWriter
from src.schemas import TrackResult


class ResultWriterTests(unittest.TestCase):
    def test_jsonl_contains_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = OutputConfig(
                directory=root,
                save_video=False,
                save_jsonl=True,
            )
            track = TrackResult(
                track_id=4,
                class_id=2,
                class_name="car",
                confidence=0.87654321,
                bbox=(1.25, 2.5, 30.0, 40.0),
                frame_id=8,
                timestamp=0.32,
            )

            with ResultWriter(config, source_fps=25.0) as writer:
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8), [track])

            record = json.loads((root / "tracks.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(
                set(record),
                {
                    "track_id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "bbox",
                    "frame_id",
                    "timestamp",
                },
            )
            self.assertEqual(record["track_id"], 4)
            self.assertEqual(record["bbox"], [1.25, 2.5, 30.0, 40.0])

    def test_result_video_can_be_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = OutputConfig(
                directory=root,
                save_video=True,
                save_jsonl=False,
            )
            with ResultWriter(config, source_fps=10.0) as writer:
                for value in (20, 80, 140):
                    frame = np.full((48, 64, 3), value, dtype=np.uint8)
                    writer.write(frame, [])

            video_path = root / "tracked.mp4"
            self.assertTrue(video_path.is_file())
            self.assertGreater(video_path.stat().st_size, 0)
            capture = cv2.VideoCapture(str(video_path))
            try:
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (48, 64))
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()

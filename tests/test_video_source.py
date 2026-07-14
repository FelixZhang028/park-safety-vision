from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.video_source import SourceError, VideoSource


class VideoSourceTests(unittest.TestCase):
    def test_local_video_frames_have_ids_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "input.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("MJPG video encoder is unavailable")
            for value in (30, 60, 90):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()

            with VideoSource(str(video_path)) as source:
                packets = list(source.frames())
                self.assertEqual(source.width, 64)
                self.assertEqual(source.height, 48)
                self.assertAlmostEqual(source.fps, 10.0, places=1)

            self.assertEqual([packet.frame_id for packet in packets], [0, 1, 2])
            self.assertEqual(packets[0].timestamp, 0.0)
            self.assertGreater(packets[-1].timestamp, packets[0].timestamp)

    def test_missing_video_has_clear_error(self) -> None:
        with self.assertRaisesRegex(SourceError, "does not exist"):
            VideoSource("definitely-missing-video.mp4").open()


if __name__ == "__main__":
    unittest.main()

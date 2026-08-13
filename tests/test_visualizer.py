from __future__ import annotations

import unittest

import numpy as np

from src.schemas import TrackResult
from src.visualizer import Visualizer


class VisualizerTests(unittest.TestCase):
    def test_annotation_preserves_frame_shape_and_draws_content(self) -> None:
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        tracks = [
            TrackResult(
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(50.0, 30.0, 100.0, 110.0),
                frame_id=0,
                timestamp=0.0,
            )
        ]

        annotated = Visualizer().annotate(frame, tracks, fps=25.0)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(np.count_nonzero(annotated), 0)
        self.assertEqual(np.count_nonzero(frame), 0)

    def test_output_width_can_upscale_presentation_video(self) -> None:
        frame = np.zeros((120, 240, 3), dtype=np.uint8)

        annotated = Visualizer(output_width=480).annotate(frame, [], fps=25.0)

        self.assertEqual(annotated.shape, (240, 480, 3))

    def test_fixed_output_canvas_preserves_aspect_ratio(self) -> None:
        frame = np.full((120, 120, 3), 255, dtype=np.uint8)

        annotated = Visualizer(output_width=320, output_height=180).annotate(
            frame, [], fps=25.0
        )

        self.assertEqual(annotated.shape, (180, 320, 3))
        self.assertEqual(np.count_nonzero(annotated[:, :60]), 0)


if __name__ == "__main__":
    unittest.main()

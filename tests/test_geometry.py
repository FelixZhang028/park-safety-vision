from __future__ import annotations

import unittest

from src.spatial.geometry import (
    bbox_bottom_center,
    line_side_distance,
    point_in_polygon,
    segments_intersect,
)


class GeometryTests(unittest.TestCase):
    def test_bottom_center_is_normalized(self) -> None:
        point = bbox_bottom_center((20.0, 10.0, 40.0, 80.0), (100, 100))
        self.assertEqual(point, (0.3, 0.8))

    def test_polygon_includes_boundary(self) -> None:
        polygon = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
        self.assertTrue(point_in_polygon((0.5, 0.5), polygon))
        self.assertTrue(point_in_polygon((0.1, 0.5), polygon))
        self.assertFalse(point_in_polygon((0.95, 0.5), polygon))

    def test_line_side_and_segment_intersection(self) -> None:
        start, end = (0.0, 0.5), (1.0, 0.5)
        self.assertLess(line_side_distance((0.5, 0.4), start, end), 0)
        self.assertGreater(line_side_distance((0.5, 0.6), start, end), 0)
        self.assertTrue(segments_intersect((0.5, 0.4), (0.5, 0.6), start, end))
        self.assertFalse(segments_intersect((1.1, 0.4), (1.1, 0.6), start, end))


if __name__ == "__main__":
    unittest.main()

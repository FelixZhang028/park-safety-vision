from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


Point = tuple[float, float]
EPSILON = 1e-9


def bbox_bottom_center(
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int],
) -> Point:
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be greater than zero")
    x1, _, x2, y2 = bbox
    x = ((x1 + x2) / 2.0) / width
    y = y2 / height
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        if _point_on_segment(point, start, end):
            return True

    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def line_side_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        raise ValueError("line endpoints must be different")
    cross = dx * (point[1] - start[1]) - dy * (point[0] - start[0])
    return cross / length


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    orientations = (
        _orientation(a1, a2, b1),
        _orientation(a1, a2, b2),
        _orientation(b1, b2, a1),
        _orientation(b1, b2, a2),
    )
    o1, o2, o3, o4 = orientations
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (abs(o1) <= EPSILON and _point_on_segment(b1, a1, a2))
        or (abs(o2) <= EPSILON and _point_on_segment(b2, a1, a2))
        or (abs(o3) <= EPSILON and _point_on_segment(a1, b1, b2))
        or (abs(o4) <= EPSILON and _point_on_segment(a2, b1, b2))
    )


def distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def polygon_to_pixels(
    polygon: Sequence[Point], frame_size: tuple[int, int]
) -> np.ndarray:
    width, height = frame_size
    points = [(int(round(x * width)), int(round(y * height))) for x, y in polygon]
    return np.asarray(points, dtype=np.int32)


def line_to_pixels(
    points: tuple[Point, Point], frame_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    width, height = frame_size
    return tuple((int(round(x * width)), int(round(y * height))) for x, y in points)


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if abs(_orientation(start, end, point)) > EPSILON:
        return False
    return (
        min(start[0], end[0]) - EPSILON <= point[0] <= max(start[0], end[0]) + EPSILON
        and min(start[1], end[1]) - EPSILON
        <= point[1]
        <= max(start[1], end[1]) + EPSILON
    )

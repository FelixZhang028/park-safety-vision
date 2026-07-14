from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..schemas import TrackResult
from ..spatial.geometry import Point, bbox_bottom_center, distance
from .schemas import TrackMotion


@dataclass(slots=True)
class _TrackHistory:
    class_name: str
    first_seen: float
    last_seen: float
    points: deque[tuple[float, Point]] = field(default_factory=deque)


class TrackHistoryStore:
    def __init__(self, history_seconds: float, stale_track_seconds: float) -> None:
        self.history_seconds = history_seconds
        self.stale_track_seconds = stale_track_seconds
        self._histories: dict[int, _TrackHistory] = {}

    def update(
        self,
        tracks: list[TrackResult],
        timestamp: float,
        frame_size: tuple[int, int],
    ) -> list[TrackMotion]:
        self._remove_stale(timestamp)
        motions: list[TrackMotion] = []
        for track in tracks:
            point = bbox_bottom_center(track.bbox, frame_size)
            history = self._histories.get(track.track_id)
            if (
                history is None
                or history.class_name != track.class_name
                or timestamp < history.last_seen
            ):
                history = _TrackHistory(
                    class_name=track.class_name,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._histories[track.track_id] = history

            previous_point = history.points[-1][1] if history.points else None
            history.points.append((timestamp, point))
            history.last_seen = timestamp
            cutoff = timestamp - self.history_seconds
            while len(history.points) > 2 and history.points[0][0] < cutoff:
                history.points.popleft()

            motions.append(
                TrackMotion(
                    track=track,
                    point=point,
                    previous_point=previous_point,
                    speed_ratio=_average_speed(history.points),
                    age_seconds=max(0.0, timestamp - history.first_seen),
                )
            )
        return motions

    def reset(self) -> None:
        self._histories.clear()

    def _remove_stale(self, timestamp: float) -> None:
        stale_ids = [
            track_id
            for track_id, history in self._histories.items()
            if timestamp - history.last_seen > self.stale_track_seconds
            or timestamp < history.last_seen
        ]
        for track_id in stale_ids:
            del self._histories[track_id]


def _average_speed(points: deque[tuple[float, Point]]) -> float:
    if len(points) < 2:
        return 0.0
    first_time, first_point = points[0]
    last_time, last_point = points[-1]
    elapsed = last_time - first_time
    if elapsed <= 1e-9:
        return 0.0
    return distance(first_point, last_point) / elapsed

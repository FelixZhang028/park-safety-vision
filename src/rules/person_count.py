from __future__ import annotations

from dataclasses import dataclass

from ..analytics.constants import PERSON_CLASS
from ..analytics.schemas import TrackMotion
from ..scene_config import CountingLineConfig, PersonCountRuleConfig, RegionConfig
from ..spatial.geometry import (
    Point,
    line_side_distance,
    point_in_polygon,
    segments_intersect,
)


@dataclass(slots=True, frozen=True)
class PersonCountStats:
    current: int
    entries: int
    exits: int


@dataclass(slots=True)
class _LineState:
    side: int
    point: Point
    last_seen: float


class PersonCounter:
    def __init__(
        self,
        config: PersonCountRuleConfig,
        regions: dict[str, RegionConfig],
        lines: dict[str, CountingLineConfig],
        stale_seconds: float,
    ) -> None:
        self.config = config
        self.regions = regions
        self.lines = lines
        self.stale_seconds = stale_seconds
        self.entries = 0
        self.exits = 0
        self._line_states: dict[tuple[int, str], _LineState] = {}
        self._last_crossing: dict[tuple[int, str], float] = {}

    def update(self, motions: list[TrackMotion], timestamp: float) -> PersonCountStats:
        people = [
            motion for motion in motions if motion.track.class_name == PERSON_CLASS
        ]
        if self.config.enabled and self.config.region:
            polygon = self.regions[self.config.region].polygon
            current = sum(point_in_polygon(motion.point, polygon) for motion in people)
            for motion in people:
                for line_name in self.config.lines:
                    self._update_line(motion, self.lines[line_name], timestamp)
        else:
            current = len(people)
        self._remove_stale(timestamp)
        return PersonCountStats(current=current, entries=self.entries, exits=self.exits)

    def reset(self) -> None:
        self.entries = 0
        self.exits = 0
        self._line_states.clear()
        self._last_crossing.clear()

    def _update_line(
        self,
        motion: TrackMotion,
        line: CountingLineConfig,
        timestamp: float,
    ) -> None:
        start, end = line.points
        key = motion.track.track_id, line.name
        previous = self._line_states.get(key)
        signed_distance = line_side_distance(motion.point, start, end)
        if abs(signed_distance) < line.hysteresis:
            if previous is not None:
                previous.last_seen = timestamp
            return
        side = 1 if signed_distance > 0 else -1
        if previous is None:
            self._line_states[key] = _LineState(side, motion.point, timestamp)
            return
        if previous.side == side:
            previous.point = motion.point
            previous.last_seen = timestamp
            return

        last_crossing = self._last_crossing.get(key, float("-inf"))
        crossed_segment = segments_intersect(previous.point, motion.point, start, end)
        if crossed_segment and timestamp - last_crossing >= line.cooldown_seconds:
            direction = (
                "negative_to_positive"
                if previous.side < side
                else "positive_to_negative"
            )
            if direction == line.in_direction:
                self.entries += 1
            else:
                self.exits += 1
            self._last_crossing[key] = timestamp
        self._line_states[key] = _LineState(side, motion.point, timestamp)

    def _remove_stale(self, timestamp: float) -> None:
        stale_keys = [
            key
            for key, state in self._line_states.items()
            if timestamp - state.last_seen > self.stale_seconds
        ]
        for key in stale_keys:
            self._line_states.pop(key, None)
            self._last_crossing.pop(key, None)

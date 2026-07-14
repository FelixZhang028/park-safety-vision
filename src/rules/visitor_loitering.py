from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from ..analytics.constants import PERSON_CLASS
from ..analytics.schemas import AlertEvent, TrackMotion
from ..events.manager import EventManager
from ..identity.schemas import TrackIdentity
from ..scene_config import (
    RegionConfig,
    VisitorLoiteringRuleConfig,
    VisitorLoiteringZoneConfig,
    clock_minutes,
)
from ..spatial.geometry import point_in_polygon


@dataclass(slots=True, frozen=True)
class VisitorLoiteringMetrics:
    current_visitors: int = 0
    pending_count: int = 0
    active_track_ids: tuple[int, ...] = ()
    active_regions: tuple[str, ...] = ()


@dataclass(slots=True)
class _DwellState:
    zone: VisitorLoiteringZoneConfig
    subject_key: str
    entered_at: float
    last_seen_at: float
    track_id: int
    identity: TrackIdentity


class VisitorLoiteringRule:
    PREFIX = "visitor_loitering:"

    def __init__(
        self,
        config: VisitorLoiteringRuleConfig,
        regions: dict[str, RegionConfig],
        scene_id: str,
    ) -> None:
        self.config = config
        self.regions = regions
        self.scene_id = scene_id
        self._states: dict[tuple[str, str], _DwellState] = {}
        self._day_start = clock_minutes(config.day_start)
        self._night_start = clock_minutes(config.night_start)

    def update(
        self,
        motions: list[TrackMotion],
        identities: Mapping[int, TrackIdentity],
        manager: EventManager,
        timestamp: float,
        frame_id: int,
        observed_at: datetime | None = None,
    ) -> tuple[VisitorLoiteringMetrics, list[AlertEvent]]:
        if not self.config.enabled:
            return VisitorLoiteringMetrics(), []

        period = self._period(observed_at)
        events: list[AlertEvent] = []
        seen_states: set[tuple[str, str]] = set()
        visitor_subjects: set[str] = set()

        for motion in motions:
            if motion.track.class_name != PERSON_CLASS:
                continue
            identity = identities.get(
                motion.track.track_id,
                TrackIdentity(track_id=motion.track.track_id),
            )
            if identity.role not in self.config.include_roles:
                continue
            visitor_subjects.add(identity.subject_key)

            for zone in self.config.zones:
                if not point_in_polygon(
                    motion.point, self.regions[zone.region].polygon
                ):
                    continue
                state_key = zone.region, identity.subject_key
                seen_states.add(state_key)
                state = self._states.get(state_key)
                event_key = self._event_key(zone.region, identity.subject_key)
                if (
                    state is None
                    or timestamp < state.last_seen_at
                    or (
                        timestamp - state.last_seen_at > zone.absence_grace_seconds
                        and not manager.is_active(event_key)
                    )
                ):
                    state = _DwellState(
                        zone=zone,
                        subject_key=identity.subject_key,
                        entered_at=timestamp,
                        last_seen_at=timestamp,
                        track_id=motion.track.track_id,
                        identity=identity,
                    )
                    self._states[state_key] = state
                else:
                    state.last_seen_at = timestamp
                    state.track_id = motion.track.track_id
                    state.identity = identity

                threshold = self._threshold(zone, period)
                dwell_seconds = max(0.0, timestamp - state.entered_at)
                events.extend(
                    manager.evaluate(
                        key=event_key,
                        event_type="visitor_abnormal_stay",
                        condition=dwell_seconds + 1e-9 >= threshold,
                        severity="critical" if period == "night" else "warning",
                        scene_id=self.scene_id,
                        region=zone.region,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        hold_seconds=0.0,
                        recovery_seconds=zone.recovery_seconds,
                        track_ids=(motion.track.track_id,),
                        details=self._details(state, period, dwell_seconds, threshold),
                    )
                )

        events.extend(
            self._update_absent_states(
                seen_states, manager, timestamp, frame_id, period
            )
        )
        active_track_ids = manager.active_track_ids(self.PREFIX)
        active_regions = tuple(
            sorted(
                {
                    state.zone.region
                    for state in self._states.values()
                    if manager.is_active(
                        self._event_key(state.zone.region, state.subject_key)
                    )
                }
            )
        )
        pending_count = sum(
            timestamp - state.last_seen_at <= state.zone.absence_grace_seconds
            and not manager.is_active(
                self._event_key(state.zone.region, state.subject_key)
            )
            for state in self._states.values()
        )
        return (
            VisitorLoiteringMetrics(
                current_visitors=len(visitor_subjects),
                pending_count=pending_count,
                active_track_ids=active_track_ids,
                active_regions=active_regions,
            ),
            events,
        )

    def reset(self) -> None:
        self._states.clear()

    def _update_absent_states(
        self,
        seen_states: set[tuple[str, str]],
        manager: EventManager,
        timestamp: float,
        frame_id: int,
        period: str,
    ) -> list[AlertEvent]:
        events: list[AlertEvent] = []
        for state_key, state in list(self._states.items()):
            if state_key in seen_states:
                continue
            if timestamp < state.last_seen_at:
                self._states.pop(state_key, None)
                continue
            if timestamp - state.last_seen_at <= state.zone.absence_grace_seconds:
                continue

            event_key = self._event_key(state.zone.region, state.subject_key)
            if manager.is_active(event_key):
                threshold = self._threshold(state.zone, period)
                events.extend(
                    manager.evaluate(
                        key=event_key,
                        event_type="visitor_abnormal_stay",
                        condition=False,
                        severity="critical" if period == "night" else "warning",
                        scene_id=self.scene_id,
                        region=state.zone.region,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        hold_seconds=0.0,
                        recovery_seconds=state.zone.recovery_seconds,
                        track_ids=(state.track_id,),
                        details=self._details(
                            state,
                            period,
                            max(0.0, state.last_seen_at - state.entered_at),
                            threshold,
                        ),
                    )
                )
            if not manager.is_active(event_key):
                self._states.pop(state_key, None)
        return events

    def _period(self, observed_at: datetime | None) -> str:
        if self.config.period != "auto":
            return self.config.period
        now = observed_at or datetime.now().astimezone()
        current = now.hour * 60 + now.minute
        if self._day_start < self._night_start:
            is_day = self._day_start <= current < self._night_start
        else:
            is_day = current >= self._day_start or current < self._night_start
        return "day" if is_day else "night"

    @staticmethod
    def _threshold(zone: VisitorLoiteringZoneConfig, period: str) -> float:
        return (
            zone.day_hold_seconds
            if period == "day"
            else zone.night_hold_seconds
        )

    @staticmethod
    def _details(
        state: _DwellState,
        period: str,
        dwell_seconds: float,
        threshold: float,
    ) -> dict:
        return {
            "person_id": state.identity.person_id,
            "role": state.identity.role,
            "identity_confidence": round(state.identity.confidence, 6),
            "accompanied": state.identity.accompanied,
            "subject_key": state.subject_key,
            "period": period,
            "entered_at": round(state.entered_at, 6),
            "dwell_seconds": round(dwell_seconds, 6),
            "threshold_seconds": round(threshold, 6),
        }

    def _event_key(self, region: str, subject_key: str) -> str:
        return f"{self.PREFIX}{region}:{subject_key}"

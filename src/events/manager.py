from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..analytics.schemas import AlertEvent


@dataclass(slots=True)
class _TemporalState:
    event_type: str
    severity: str
    scene_id: str
    region: str
    hold_seconds: float
    recovery_seconds: float
    condition_since: float | None = None
    active_since: float | None = None
    recovery_since: float | None = None
    event_id: str | None = None
    track_ids: tuple[int, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


class EventManager:
    def __init__(self) -> None:
        self._states: dict[str, _TemporalState] = {}

    def evaluate(
        self,
        *,
        key: str,
        event_type: str,
        condition: bool,
        severity: str,
        scene_id: str,
        region: str,
        timestamp: float,
        frame_id: int,
        hold_seconds: float,
        recovery_seconds: float,
        track_ids: tuple[int, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> list[AlertEvent]:
        state = self._states.get(key)
        if state is None:
            if not condition:
                return []
            state = _TemporalState(
                event_type=event_type,
                severity=severity,
                scene_id=scene_id,
                region=region,
                hold_seconds=hold_seconds,
                recovery_seconds=recovery_seconds,
            )
            self._states[key] = state

        state.track_ids = track_ids
        state.details = dict(details or {})
        return self._advance(key, state, condition, timestamp, frame_id)

    def mark_missing(
        self,
        prefix: str,
        seen_keys: set[str],
        timestamp: float,
        frame_id: int,
    ) -> list[AlertEvent]:
        events: list[AlertEvent] = []
        for key in [
            candidate
            for candidate in self._states
            if candidate.startswith(prefix) and candidate not in seen_keys
        ]:
            state = self._states.get(key)
            if state is not None:
                events.extend(self._advance(key, state, False, timestamp, frame_id))
        return events

    def is_active(self, key: str) -> bool:
        state = self._states.get(key)
        return state is not None and state.active_since is not None

    def active_track_ids(self, prefix: str) -> tuple[int, ...]:
        track_ids = {
            track_id
            for key, state in self._states.items()
            if key.startswith(prefix) and state.active_since is not None
            for track_id in state.track_ids
        }
        return tuple(sorted(track_ids))

    def active_event_types(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    state.event_type
                    for state in self._states.values()
                    if state.active_since is not None
                }
            )
        )

    def reset(self) -> None:
        self._states.clear()

    def _advance(
        self,
        key: str,
        state: _TemporalState,
        condition: bool,
        timestamp: float,
        frame_id: int,
    ) -> list[AlertEvent]:
        if condition:
            state.recovery_since = None
            if state.active_since is not None:
                return []
            if state.condition_since is None:
                state.condition_since = timestamp
            if timestamp - state.condition_since + 1e-9 < state.hold_seconds:
                return []

            state.active_since = state.condition_since
            state.event_id = uuid4().hex
            return [self._event(state, "started", timestamp, frame_id)]

        state.condition_since = None
        if state.active_since is None:
            self._states.pop(key, None)
            return []
        if state.recovery_since is None:
            state.recovery_since = timestamp
        if timestamp - state.recovery_since + 1e-9 < state.recovery_seconds:
            return []

        details = dict(state.details)
        details["duration_seconds"] = round(timestamp - state.active_since, 6)
        state.details = details
        event = self._event(state, "ended", timestamp, frame_id)
        self._states.pop(key, None)
        return [event]

    @staticmethod
    def _event(
        state: _TemporalState,
        event_state: str,
        timestamp: float,
        frame_id: int,
    ) -> AlertEvent:
        assert state.event_id is not None
        return AlertEvent(
            event_id=state.event_id,
            event_type=state.event_type,
            state=event_state,
            severity=state.severity,
            scene_id=state.scene_id,
            region=state.region,
            timestamp=timestamp,
            frame_id=frame_id,
            track_ids=state.track_ids,
            details=dict(state.details),
        )

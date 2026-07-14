from __future__ import annotations

from dataclasses import dataclass


IDENTITY_ROLES = frozenset({"employee", "visitor", "unknown"})


@dataclass(slots=True, frozen=True)
class TrackIdentity:
    track_id: int
    person_id: str | None = None
    role: str = "unknown"
    confidence: float = 0.0
    accompanied: bool | None = None

    @property
    def subject_key(self) -> str:
        return f"person:{self.person_id}" if self.person_id else f"track:{self.track_id}"

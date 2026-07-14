from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..analytics.constants import PERSON_CLASS
from ..schemas import TrackResult
from .schemas import TrackIdentity


class UnknownIdentityProvider:
    """Fallback provider used until a face or person ReID service is configured."""

    def identify(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackResult],
    ) -> dict[int, TrackIdentity]:
        del frame
        return {
            track.track_id: TrackIdentity(track_id=track.track_id)
            for track in tracks
            if track.class_name == PERSON_CLASS
        }

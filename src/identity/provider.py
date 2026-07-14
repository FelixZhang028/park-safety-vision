from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..schemas import TrackResult
from .schemas import TrackIdentity


class IdentityProvider(Protocol):
    def identify(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackResult],
    ) -> dict[int, TrackIdentity]: ...

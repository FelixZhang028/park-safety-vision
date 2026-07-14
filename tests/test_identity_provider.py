from __future__ import annotations

import unittest

import numpy as np

from src.identity import UnknownIdentityProvider
from src.schemas import TrackResult


def make_track(track_id: int, class_name: str) -> TrackResult:
    return TrackResult(
        track_id=track_id,
        class_id=0 if class_name == "person" else 2,
        class_name=class_name,
        confidence=0.9,
        bbox=(10.0, 10.0, 20.0, 30.0),
        frame_id=0,
        timestamp=0.0,
    )


class UnknownIdentityProviderTests(unittest.TestCase):
    def test_only_people_receive_unknown_identity(self) -> None:
        identities = UnknownIdentityProvider().identify(
            np.zeros((40, 40, 3), dtype=np.uint8),
            [make_track(1, "person"), make_track(2, "car")],
        )

        self.assertEqual(set(identities), {1})
        self.assertEqual(identities[1].role, "unknown")
        self.assertEqual(identities[1].subject_key, "track:1")


if __name__ == "__main__":
    unittest.main()

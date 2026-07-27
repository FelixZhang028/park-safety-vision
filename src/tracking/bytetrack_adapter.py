from __future__ import annotations

from dataclasses import dataclass

from ..config import TrackingConfig
from ..inference.schemas import DetectionResult
from ..schemas import TrackResult


@dataclass(slots=True)
class _TrackState:
    track_id: int
    detection: DetectionResult
    missed_frames: int = 0


class ByteTrackAdapter:
    """Dependency-light two-stage association for RK3588 and desktop parity."""

    def __init__(self, config: TrackingConfig, high_threshold: float) -> None:
        self.config = config
        self.high_threshold = high_threshold
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[DetectionResult],
        frame_id: int,
        timestamp: float,
    ) -> list[TrackResult]:
        if not self.config.persist:
            self.reset()
        for state in self._tracks.values():
            state.missed_frames += 1

        high = [item for item in detections if item.confidence >= self.high_threshold]
        low = [
            item
            for item in detections
            if self.config.track_low_threshold <= item.confidence < self.high_threshold
        ]
        active_ids = set(self._tracks)
        matched: dict[int, DetectionResult] = {}

        first_matches, active_ids, unmatched_high = _associate(
            self._tracks,
            active_ids,
            high,
            self.config.match_iou_threshold,
        )
        matched.update(first_matches)
        second_matches, _, _ = _associate(
            self._tracks,
            active_ids,
            low,
            max(0.10, self.config.match_iou_threshold * 0.75),
        )
        matched.update(second_matches)

        for track_id, detection in matched.items():
            state = self._tracks[track_id]
            state.detection = detection
            state.missed_frames = 0

        for detection in unmatched_high:
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(track_id, detection)
            matched[track_id] = detection

        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if state.missed_frames > self.config.track_buffer_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

        return [
            TrackResult(
                track_id=track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            for track_id, detection in sorted(matched.items())
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1


def _associate(
    tracks: dict[int, _TrackState],
    candidate_track_ids: set[int],
    detections: list[DetectionResult],
    threshold: float,
) -> tuple[dict[int, DetectionResult], set[int], list[DetectionResult]]:
    available_tracks = set(candidate_track_ids)
    available_detections = set(range(len(detections)))
    candidates: list[tuple[float, int, int]] = []
    for track_id in available_tracks:
        previous = tracks[track_id].detection
        for detection_index, detection in enumerate(detections):
            if previous.class_id != detection.class_id:
                continue
            overlap = _bbox_iou(previous.bbox, detection.bbox)
            if overlap >= threshold:
                candidates.append((overlap, track_id, detection_index))

    matches: dict[int, DetectionResult] = {}
    for _, track_id, detection_index in sorted(candidates, reverse=True):
        if track_id not in available_tracks or detection_index not in available_detections:
            continue
        matches[track_id] = detections[detection_index]
        available_tracks.remove(track_id)
        available_detections.remove(detection_index)

    unmatched = [detections[index] for index in sorted(available_detections)]
    return matches, available_tracks, unmatched


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0

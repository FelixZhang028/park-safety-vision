from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..analytics.constants import PERSON_CLASS, VEHICLE_CLASSES
from ..config import ConfigurationError
from ..scene_config import FireLaneObstructionRuleConfig, RegionConfig
from ..schemas import TrackResult
from ..spatial.geometry import polygon_to_pixels
from .schemas import ObstructionCandidate, ObstructionDetection


class BackgroundObstructionDetector:
    def __init__(
        self,
        config: FireLaneObstructionRuleConfig,
        region: RegionConfig,
    ) -> None:
        if config.baseline_path is None:
            raise ConfigurationError(
                "fire_lane_obstruction.baseline_path cannot be empty"
            )
        self.config = config
        self.region = region
        self._baseline_source = _read_image(config.baseline_path)
        self._frame_size: tuple[int, int] | None = None
        self._baseline_gray: np.ndarray | None = None
        self._region_mask: np.ndarray | None = None
        self._region_area = 0
        self._persistence: np.ndarray | None = None
        self._last_timestamp: float | None = None

    def update(
        self,
        frame: np.ndarray,
        tracks: list[TrackResult],
        timestamp: float,
    ) -> ObstructionDetection:
        self._prepare(frame)
        assert self._baseline_gray is not None
        assert self._region_mask is not None
        assert self._persistence is not None

        if self._last_timestamp is None or timestamp < self._last_timestamp:
            self._persistence.fill(0.0)
            elapsed = 0.0
        else:
            elapsed = timestamp - self._last_timestamp
        self._last_timestamp = timestamp

        current = _gray_blurred(frame)
        difference = cv2.absdiff(self._baseline_gray, current)
        changed = cv2.threshold(
            difference,
            self.config.pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]
        changed = cv2.bitwise_and(changed, self._region_mask)
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_CLOSE,
            np.ones((_close_kernel(frame),) * 2, dtype=np.uint8),
        )
        self._exclude_tracked_objects(changed, tracks)
        raw_change_ratio = cv2.countNonZero(changed) / max(1, self._region_area)

        if raw_change_ratio >= self.config.max_global_change_ratio:
            self._persistence.fill(0.0)
            return ObstructionDetection(
                raw_change_ratio=raw_change_ratio,
                scene_change_detected=True,
            )

        active = changed > 0
        self._persistence[active] = np.minimum(
            self._persistence[active] + elapsed,
            self.config.hold_seconds + 1.0,
        )
        self._persistence[~active] = np.maximum(
            0.0,
            self._persistence[~active] - elapsed * 2.0,
        )

        stable = (
            changed.copy()
            if self.config.hold_seconds <= 0
            else (
                self._persistence + 1e-9 >= self.config.hold_seconds
            ).astype(np.uint8) * 255
        )
        stable = cv2.morphologyEx(
            stable,
            cv2.MORPH_CLOSE,
            np.ones((_stable_kernel(frame),) * 2, dtype=np.uint8),
        )
        candidates = self._candidates(stable, frame.shape[1], frame.shape[0])
        return ObstructionDetection(
            candidates=candidates,
            changed_area_ratio=sum(candidate.area_ratio for candidate in candidates),
            raw_change_ratio=raw_change_ratio,
        )

    def reset(self) -> None:
        if self._persistence is not None:
            self._persistence.fill(0.0)
        self._last_timestamp = None

    def _prepare(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        frame_size = width, height
        if self._frame_size == frame_size:
            return
        baseline = self._baseline_source
        if baseline.shape[:2] != (height, width):
            baseline = cv2.resize(baseline, frame_size, interpolation=cv2.INTER_AREA)
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(
            region_mask,
            [polygon_to_pixels(self.region.polygon, frame_size)],
            255,
        )
        self._frame_size = frame_size
        self._baseline_gray = _gray_blurred(baseline)
        self._region_mask = region_mask
        self._region_area = cv2.countNonZero(region_mask)
        self._persistence = np.zeros((height, width), dtype=np.float32)
        self._last_timestamp = None

    def _exclude_tracked_objects(
        self,
        changed: np.ndarray,
        tracks: list[TrackResult],
    ) -> None:
        assert self._persistence is not None
        height, width = changed.shape
        padding_x = int(round(width * self.config.exclusion_padding_ratio))
        padding_y = int(round(height * self.config.exclusion_padding_ratio))
        for track in tracks:
            if (
                track.class_name != PERSON_CLASS
                and track.class_name not in VEHICLE_CLASSES
            ):
                continue
            x1, y1, x2, y2 = track.bbox
            left = max(0, int(x1) - padding_x)
            top = max(0, int(y1) - padding_y)
            right = min(width, int(np.ceil(x2)) + padding_x)
            bottom = min(height, int(np.ceil(y2)) + padding_y)
            changed[top:bottom, left:right] = 0
            self._persistence[top:bottom, left:right] = 0.0

    def _candidates(
        self,
        stable: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[ObstructionCandidate, ...]:
        contours, _ = cv2.findContours(
            stable,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: list[ObstructionCandidate] = []
        for contour in contours:
            area_ratio = cv2.contourArea(contour) / max(1, self._region_area)
            if area_ratio < self.config.min_area_ratio:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            candidates.append(
                ObstructionCandidate(
                    bbox=(
                        x / width,
                        y / height,
                        (x + box_width) / width,
                        (y + box_height) / height,
                    ),
                    area_ratio=area_ratio,
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.area_ratio, reverse=True))


def _read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read obstruction baseline: {path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigurationError(f"Invalid obstruction baseline image: {path}")
    return image


def _gray_blurred(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)


def _odd_kernel(value: int) -> int:
    value = max(3, value)
    return value if value % 2 else value + 1


def _close_kernel(frame: np.ndarray) -> int:
    return _odd_kernel(int(round(min(frame.shape[:2]) * 0.02)))


def _stable_kernel(frame: np.ndarray) -> int:
    return _odd_kernel(int(round(min(frame.shape[:2]) * 0.015)))

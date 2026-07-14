from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import cv2

from .schemas import FramePacket


LOGGER = logging.getLogger(__name__)
STREAM_SCHEMES = {"rtsp", "rtsps", "rtmp", "http", "https"}


class SourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    def __init__(
        self,
        source: str | int,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        self.original_source = source
        self.source = _normalize_source(source)
        self.kind = _source_kind(self.source)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.capture: cv2.VideoCapture | None = None
        self._live_started_at = 0.0
        self._fps = 25.0

    def validate(self) -> None:
        if self.kind == "file":
            source_path = Path(str(self.source)).expanduser()
            if not source_path.is_file():
                raise SourceError(f"Video file does not exist: {source_path.resolve()}")

    def open(self) -> "VideoSource":
        if self.kind == "file":
            source_path = Path(str(self.source)).expanduser()
            if not source_path.is_file():
                raise SourceError(f"Video file does not exist: {source_path.resolve()}")
            self.source = str(source_path.resolve())

        self.capture = self._create_capture()
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise SourceError(f"Cannot open {self.kind} source: {self.original_source}")

        source_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if math.isfinite(source_fps) and source_fps > 0:
            self._fps = source_fps
        self._live_started_at = time.perf_counter()
        return self

    def _create_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.source)
        if self.kind in {"camera", "stream"}:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        if self.capture is None:
            return 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self.capture is None:
            return 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int | None:
        if self.capture is None or self.kind != "file":
            return None
        value = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return value if value > 0 else None

    def frames(self) -> Iterator[FramePacket]:
        if self.capture is None:
            raise SourceError("Video source has not been opened")

        frame_id = 0
        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                if self.kind == "file":
                    break
                frame = self._read_after_reconnect()

            timestamp = self._timestamp(frame_id)
            yield FramePacket(frame=frame, frame_id=frame_id, timestamp=timestamp)
            frame_id += 1

    def _read_after_reconnect(self):
        for attempt in range(1, self.reconnect_attempts + 1):
            LOGGER.warning(
                "Source interrupted; reconnecting (%d/%d)",
                attempt,
                self.reconnect_attempts,
            )
            self.release()
            if self.reconnect_delay_seconds:
                time.sleep(self.reconnect_delay_seconds)
            self.capture = self._create_capture()
            if not self.capture.isOpened():
                continue
            ok, frame = self.capture.read()
            if ok and frame is not None:
                LOGGER.info("Source reconnected")
                return frame
        raise SourceError(
            f"Source interrupted and reconnect failed: {self.original_source}"
        )

    def _timestamp(self, frame_id: int) -> float:
        if self.capture is None:
            return frame_id / self._fps
        if self.kind == "file":
            position_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            if math.isfinite(position_ms) and position_ms > 0:
                return position_ms / 1000.0
            return frame_id / self._fps
        return time.perf_counter() - self._live_started_at

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def _normalize_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    stripped = source.strip()
    if stripped.isdecimal():
        return int(stripped)
    return stripped


def _source_kind(source: str | int) -> str:
    if isinstance(source, int):
        return "camera"
    if urlparse(source).scheme.lower() in STREAM_SCHEMES:
        return "stream"
    return "file"

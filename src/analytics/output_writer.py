from __future__ import annotations

import json
from collections import Counter
from typing import TextIO

import cv2
import numpy as np

from ..config import OutputConfig
from .schemas import AlertEvent, AnalysisResult, AnalyticsSnapshot


class AnalyticsOutputWriter:
    def __init__(
        self,
        config: OutputConfig,
        metrics_interval_seconds: float,
    ) -> None:
        self.config = config
        self.metrics_interval_seconds = metrics_interval_seconds
        self.metrics_path = config.directory / config.metrics_name
        self.events_path = config.directory / config.events_name
        self.summary_path = config.directory / config.summary_name
        self.event_images_directory = config.directory / config.event_images_directory
        self._metrics_file: TextIO | None = None
        self._events_file: TextIO | None = None
        self._last_metrics_timestamp: float | None = None
        self._last_snapshot: AnalyticsSnapshot | None = None
        self._processed_frames = 0
        self._event_counts: Counter[str] = Counter()

    def open(self) -> None:
        self.event_images_directory.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self.metrics_path.open("w", encoding="utf-8", buffering=1)
        self._events_file = self.events_path.open("w", encoding="utf-8", buffering=1)

    def write(self, frame: np.ndarray, analysis: AnalysisResult) -> None:
        self._last_snapshot = analysis.snapshot
        if self._should_write_metrics(analysis.snapshot.timestamp):
            assert self._metrics_file is not None
            self._write_json_line(self._metrics_file, analysis.snapshot.to_dict())
            self._last_metrics_timestamp = analysis.snapshot.timestamp

        assert self._events_file is not None
        for event in analysis.events:
            if event.state == "started":
                self._save_event_image(frame, event)
                self._event_counts[event.event_type] += 1
            self._write_json_line(self._events_file, event.to_dict())
        self._processed_frames += 1

    def close(self) -> None:
        self._write_summary()
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None

    def _should_write_metrics(self, timestamp: float) -> bool:
        return (
            self._last_metrics_timestamp is None
            or timestamp < self._last_metrics_timestamp
            or timestamp - self._last_metrics_timestamp
            >= self.metrics_interval_seconds - 1e-9
        )

    def _save_event_image(self, frame: np.ndarray, event: AlertEvent) -> None:
        milliseconds = int(round(event.timestamp * 1000))
        file_name = f"{milliseconds:012d}_{event.event_type}_{event.event_id[:8]}.jpg"
        image_path = self.event_images_directory / file_name
        if not cv2.imwrite(str(image_path), frame):
            raise OSError(f"Cannot write event image: {image_path}")
        event.snapshot_path = image_path.relative_to(self.config.directory).as_posix()

    def _write_summary(self) -> None:
        summary = {
            "processed_frames": self._processed_frames,
            "event_counts": dict(sorted(self._event_counts.items())),
            "final_metrics": (
                self._last_snapshot.to_dict()
                if self._last_snapshot is not None
                else None
            ),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_json_line(file: TextIO, data: dict) -> None:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")

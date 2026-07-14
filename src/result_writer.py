from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from .analytics.output_writer import AnalyticsOutputWriter
from .analytics.schemas import AnalysisResult
from .config import OutputConfig
from .schemas import TrackResult


class OutputError(RuntimeError):
    """Raised when an output file cannot be created or written."""


class ResultWriter:
    def __init__(
        self,
        config: OutputConfig,
        source_fps: float,
        analytics_enabled: bool = False,
        metrics_interval_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.source_fps = source_fps if source_fps > 0 else 25.0
        self.video_path = (
            config.directory / config.video_name if config.save_video else None
        )
        self.jsonl_path = (
            config.directory / config.jsonl_name if config.save_jsonl else None
        )
        self._video_writer: cv2.VideoWriter | None = None
        self._video_size: tuple[int, int] | None = None
        self._jsonl_file: TextIO | None = None
        self._analytics_writer = (
            AnalyticsOutputWriter(config, metrics_interval_seconds)
            if analytics_enabled
            else None
        )

    def open(self) -> "ResultWriter":
        try:
            self.config.directory.mkdir(parents=True, exist_ok=True)
            if self.jsonl_path is not None:
                self._jsonl_file = self.jsonl_path.open(
                    "w", encoding="utf-8", buffering=1
                )
            if self._analytics_writer is not None:
                self._analytics_writer.open()
        except OSError as exc:
            raise OutputError(f"Cannot create output files: {exc}") from exc
        return self

    def write(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackResult],
        analysis: AnalysisResult | None = None,
    ) -> None:
        if self.video_path is not None:
            self._write_video_frame(frame)
        if self._jsonl_file is not None:
            self._write_tracks(tracks)
        if analysis is not None and self._analytics_writer is not None:
            try:
                self._analytics_writer.write(frame, analysis)
            except OSError as exc:
                raise OutputError(f"Cannot write analytics output: {exc}") from exc

    def _write_video_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        frame_size = (width, height)
        if self._video_writer is None:
            self._open_video_writer(frame_size)
        elif self._video_size != frame_size:
            raise OutputError(
                f"Frame size changed from {self._video_size} to {frame_size}"
            )

        assert self._video_writer is not None
        self._video_writer.write(frame)

    def _open_video_writer(self, frame_size: tuple[int, int]) -> None:
        assert self.video_path is not None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.video_path), fourcc, self.source_fps, frame_size
        )
        if not writer.isOpened():
            writer.release()
            raise OutputError(f"Cannot open result video: {self.video_path}")
        self._video_writer = writer
        self._video_size = frame_size

    def _write_tracks(self, tracks: Sequence[TrackResult]) -> None:
        assert self._jsonl_file is not None
        try:
            for track in tracks:
                self._jsonl_file.write(
                    json.dumps(track.to_dict(), ensure_ascii=False) + "\n"
                )
        except OSError as exc:
            raise OutputError(f"Cannot write JSONL output: {exc}") from exc

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._analytics_writer is not None:
            try:
                self._analytics_writer.close()
            except OSError as exc:
                raise OutputError(f"Cannot close analytics output: {exc}") from exc

    def __enter__(self) -> "ResultWriter":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def output_paths(config: OutputConfig) -> tuple[Path | None, Path | None]:
    video_path = config.directory / config.video_name if config.save_video else None
    jsonl_path = config.directory / config.jsonl_name if config.save_jsonl else None
    return video_path, jsonl_path

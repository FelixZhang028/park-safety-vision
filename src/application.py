from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .analytics.engine import AnalyticsEngine
from .config import AppConfig
from .detector_tracker import DetectorTracker
from .identity import UnknownIdentityProvider
from .result_writer import ResultWriter, output_paths
from .scene_config import SceneConfig
from .video_source import SourceError, VideoSource
from .visualizer import Visualizer


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RunSummary:
    total_frames: int
    elapsed_seconds: float
    average_fps: float
    device: str
    video_path: Path | None
    jsonl_path: Path | None
    metrics_path: Path | None
    events_path: Path | None
    summary_path: Path | None


def run_tracking(
    config: AppConfig,
    source: str | int,
    max_frames: int | None = None,
    scene: SceneConfig | None = None,
) -> RunSummary:
    source_reader = VideoSource(
        source,
        reconnect_attempts=config.runtime.reconnect_attempts,
        reconnect_delay_seconds=config.runtime.reconnect_delay_seconds,
    )
    analytics = AnalyticsEngine(scene) if scene is not None else None
    identity_provider = UnknownIdentityProvider() if scene is not None else None
    source_reader.validate()
    detector = DetectorTracker(config.model, config.tracking)
    metrics_path = (
        config.output.directory / config.output.metrics_name if scene else None
    )
    events_path = config.output.directory / config.output.events_name if scene else None
    summary_path = (
        config.output.directory / config.output.summary_name if scene else None
    )
    visualizer = Visualizer(config.display.line_width)
    video_path, jsonl_path = output_paths(config.output)
    total_frames = 0
    smoothed_fps = 0.0
    started_at = 0.0
    show_window = config.display.show_window

    LOGGER.info("Device: %s", detector.device_description)
    LOGGER.info("Model: %s", config.model.path)
    LOGGER.info("Source: %s", source)

    try:
        with (
            source_reader as video_source,
            ResultWriter(
                config.output,
                video_source.fps,
                analytics_enabled=analytics is not None,
                metrics_interval_seconds=(
                    scene.metrics_interval_seconds if scene is not None else 1.0
                ),
            ) as writer,
        ):
            LOGGER.info(
                "Video: %dx%d at %.2f FPS%s",
                video_source.width,
                video_source.height,
                video_source.fps,
                (
                    f", {video_source.total_frames} frames"
                    if video_source.total_frames is not None
                    else ""
                ),
            )

            started_at = time.perf_counter()
            for packet in video_source.frames():
                if max_frames is not None and total_frames >= max_frames:
                    break

                frame_started_at = time.perf_counter()
                tracks = detector.track(
                    packet.frame,
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                )
                identities = (
                    identity_provider.identify(packet.frame, tracks)
                    if identity_provider is not None
                    else {}
                )
                analysis = (
                    analytics.update(
                        tracks,
                        timestamp=packet.timestamp,
                        frame_id=packet.frame_id,
                        frame_size=(packet.frame.shape[1], packet.frame.shape[0]),
                        identities=identities,
                        frame=packet.frame,
                    )
                    if analytics is not None
                    else None
                )
                processing_seconds = max(time.perf_counter() - frame_started_at, 1e-9)
                instant_fps = 1.0 / processing_seconds
                smoothed_fps = (
                    instant_fps
                    if smoothed_fps == 0.0
                    else 0.90 * smoothed_fps + 0.10 * instant_fps
                )
                annotated = visualizer.annotate(
                    packet.frame,
                    tracks,
                    smoothed_fps,
                    analysis=analysis,
                    scene=scene,
                )
                writer.write(annotated, tracks, analysis)
                if analysis is not None:
                    for event in analysis.events:
                        LOGGER.warning(
                            "Event %s %s region=%s tracks=%s",
                            event.event_type,
                            event.state,
                            event.region,
                            event.track_ids,
                        )
                total_frames += 1

                if show_window:
                    try:
                        cv2.imshow(config.display.window_name, annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            LOGGER.info("Stopped by q key")
                            break
                    except cv2.error as exc:
                        LOGGER.warning(
                            "Preview unavailable; continuing headless: %s", exc
                        )
                        show_window = False

        if total_frames == 0:
            raise SourceError(f"No frames could be read from source: {source}")
    finally:
        detector.close()
        if config.display.show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    return RunSummary(
        total_frames=total_frames,
        elapsed_seconds=elapsed_seconds,
        average_fps=total_frames / elapsed_seconds,
        device=detector.device_description,
        video_path=video_path,
        jsonl_path=jsonl_path,
        metrics_path=metrics_path,
        events_path=events_path,
        summary_path=summary_path,
    )

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from .analytics.overlay import AnalyticsOverlay
from .analytics.schemas import AnalysisResult
from .scene_config import SceneConfig

from .schemas import TrackResult


VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
CLASS_COLORS = {
    "person": (64, 200, 64),
    "car": (235, 145, 40),
    "motorcycle": (30, 150, 245),
    "bus": (50, 210, 220),
    "truck": (210, 80, 190),
}
DEFAULT_COLOR = (210, 210, 210)


class Visualizer:
    def __init__(
        self,
        line_width: int = 2,
        presentation_mode: bool = False,
        show_fps: bool = True,
        show_track_labels: bool = True,
        output_width: int = 0,
        output_height: int = 0,
    ) -> None:
        self.line_width = line_width
        self.presentation_mode = presentation_mode
        self.show_fps = show_fps
        self.show_track_labels = show_track_labels
        self.output_width = output_width
        self.output_height = output_height
        self.analytics_overlay = AnalyticsOverlay()

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackResult],
        fps: float,
        analysis: AnalysisResult | None = None,
        scene: SceneConfig | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        snapshot = analysis.snapshot if analysis is not None else None
        if snapshot is not None and scene is not None:
            self.analytics_overlay.draw_scene(
                canvas,
                scene,
                snapshot,
                presentation_mode=self.presentation_mode,
            )

        height, width = canvas.shape[:2]

        visible_tracks = self._visible_tracks(tracks, scene)
        for track in visible_tracks:
            x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            color = CLASS_COLORS.get(track.class_name, DEFAULT_COLOR)
            if snapshot is not None:
                color = self.analytics_overlay.track_color(
                    track.track_id, snapshot, color
                )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.line_width)
            if self.show_track_labels:
                label = (
                    f"{track.class_name}  ID:{track.track_id}"
                    f"  {track.confidence:.2f}"
                )
                self._draw_label(canvas, label, x1, y1, color)

        if snapshot is not None and scene is not None and not self.presentation_mode:
            self.analytics_overlay.draw_status(
                canvas,
                fps,
                snapshot,
                scene,
                presentation_mode=self.presentation_mode,
                show_fps=self.show_fps,
            )
        elif snapshot is None or scene is None:
            person_count = sum(track.class_name == "person" for track in tracks)
            vehicle_count = sum(track.class_name in VEHICLE_CLASSES for track in tracks)
            self._draw_status(canvas, fps, person_count, vehicle_count)
        canvas = self._resize_output(canvas)
        if snapshot is not None and scene is not None and self.presentation_mode:
            self.analytics_overlay.draw_status(
                canvas,
                fps,
                snapshot,
                scene,
                presentation_mode=True,
                show_fps=self.show_fps,
            )
        return canvas

    def _visible_tracks(
        self,
        tracks: Sequence[TrackResult],
        scene: SceneConfig | None,
    ) -> Sequence[TrackResult]:
        if not self.presentation_mode or scene is None:
            return tracks

        visible_classes: set[str] = set()
        if scene.person_count.enabled or scene.visitor_loitering.enabled:
            visible_classes.add("person")
        if (
            scene.congestion.enabled
            or scene.illegal_parking.enabled
            or scene.fire_lane.enabled
        ):
            visible_classes.update(VEHICLE_CLASSES)
        return tuple(
            track for track in tracks if track.class_name in visible_classes
        )

    def _resize_output(self, frame: np.ndarray) -> np.ndarray:
        if self.output_width > 0 and self.output_height > 0:
            target_size = self.output_width, self.output_height
            if (frame.shape[1], frame.shape[0]) == target_size:
                return frame
            scale = min(
                self.output_width / frame.shape[1],
                self.output_height / frame.shape[0],
            )
            resized_size = (
                max(1, int(round(frame.shape[1] * scale))),
                max(1, int(round(frame.shape[0] * scale))),
            )
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            resized = cv2.resize(frame, resized_size, interpolation=interpolation)
            canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            left = (self.output_width - resized_size[0]) // 2
            top = (self.output_height - resized_size[1]) // 2
            canvas[top : top + resized_size[1], left : left + resized_size[0]] = resized
            return canvas
        if self.output_width <= 0 or frame.shape[1] >= self.output_width:
            return frame
        scale = self.output_width / frame.shape[1]
        size = self.output_width, int(round(frame.shape[0] * scale))
        return cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        top = max(0, y - text_height - baseline - 8)
        right = min(frame.shape[1] - 1, x + text_width + 8)
        cv2.rectangle(frame, (x, top), (right, y), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 4, max(text_height + 2, y - baseline - 4)),
            font,
            scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_status(
        frame: np.ndarray,
        fps: float,
        person_count: int,
        vehicle_count: int,
    ) -> None:
        panel_width = min(315, frame.shape[1])
        panel_height = min(72, frame.shape[0])
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Person: {person_count}    Vehicle: {vehicle_count}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

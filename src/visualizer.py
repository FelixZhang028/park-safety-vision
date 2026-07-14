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
    def __init__(self, line_width: int = 2) -> None:
        self.line_width = line_width
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
            self.analytics_overlay.draw_scene(canvas, scene, snapshot)

        height, width = canvas.shape[:2]

        for track in tracks:
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
            label = f"{track.class_name}  ID:{track.track_id}  {track.confidence:.2f}"
            self._draw_label(canvas, label, x1, y1, color)

        if snapshot is not None:
            self.analytics_overlay.draw_status(canvas, fps, snapshot)
        else:
            person_count = sum(track.class_name == "person" for track in tracks)
            vehicle_count = sum(track.class_name in VEHICLE_CLASSES for track in tracks)
            self._draw_status(canvas, fps, person_count, vehicle_count)
        return canvas

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

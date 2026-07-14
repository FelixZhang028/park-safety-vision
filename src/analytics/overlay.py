from __future__ import annotations

import cv2
import numpy as np

from ..scene_config import SceneConfig
from ..spatial.geometry import line_to_pixels, polygon_to_pixels
from .schemas import AnalyticsSnapshot


PEOPLE_REGION_COLOR = (70, 190, 70)
CONGESTION_REGION_COLOR = (30, 200, 220)
PARKING_REGION_COLOR = (20, 140, 245)
FIRE_LANE_REGION_COLOR = (220, 200, 40)
DEFAULT_REGION_COLOR = (180, 180, 180)
ALERT_COLOR = (40, 40, 235)


class AnalyticsOverlay:
    def draw_scene(
        self,
        frame: np.ndarray,
        scene: SceneConfig,
        snapshot: AnalyticsSnapshot,
    ) -> None:
        frame_size = frame.shape[1], frame.shape[0]
        styles = {}
        overlay = frame.copy()
        has_active_region = False
        for name, region in scene.regions.items():
            color, active = self._region_style(name, scene, snapshot)
            pixels = polygon_to_pixels(region.polygon, frame_size)
            if active:
                cv2.fillPoly(overlay, [pixels], ALERT_COLOR)
                has_active_region = True
            styles[name] = pixels, ALERT_COLOR if active else color

        if has_active_region:
            cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

        for name, (pixels, color) in styles.items():
            cv2.polylines(frame, [pixels], True, color, 2, cv2.LINE_AA)
            label_x, label_y = pixels[0].tolist()
            cv2.putText(
                frame,
                name,
                (max(4, label_x + 4), max(18, label_y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA,
            )

        for box in snapshot.fire_lane_obstruction_boxes:
            x1, y1, x2, y2 = box
            left = int(round(x1 * frame_size[0]))
            top = int(round(y1 * frame_size[1]))
            right = int(round(x2 * frame_size[0]))
            bottom = int(round(y2 * frame_size[1]))
            cv2.rectangle(
                frame, (left, top), (right, bottom), ALERT_COLOR, 3, cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "UNKNOWN OBSTRUCTION",
                (left, max(20, top - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ALERT_COLOR,
                2,
                cv2.LINE_AA,
            )

        for name, line in scene.lines.items():
            start, end = line_to_pixels(line.points, frame_size)
            cv2.line(frame, start, end, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                name,
                (start[0] + 4, max(18, start[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_status(
        frame: np.ndarray,
        fps: float,
        snapshot: AnalyticsSnapshot,
    ) -> None:
        panel_width = min(500, frame.shape[1])
        panel_height = min(229, frame.shape[0])
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

        congestion = "ALERT" if snapshot.congestion_active else "OK"
        parking_ids = _format_ids(snapshot.illegal_parking_track_ids)
        fire_ids = _format_ids(snapshot.fire_lane_track_ids)
        obstruction = (
            "SCENE CHANGE"
            if snapshot.fire_lane_scene_change_detected
            else "ALERT" if snapshot.fire_lane_obstruction_active else "OK"
        )
        loitering_ids = _format_ids(snapshot.visitor_loitering_track_ids)
        lines = (
            f"FPS: {fps:.1f}    Scene: {snapshot.scene_id}",
            (
                f"People: {snapshot.current_people}    In: {snapshot.entries}"
                f"    Out: {snapshot.exits}"
            ),
            (
                f"Visitors: {snapshot.current_visitors}"
                f"    Pending: {snapshot.visitor_loitering_pending}"
                f"    Stay IDs: {loitering_ids}"
            ),
            (f"Vehicles: {snapshot.current_vehicles}    Congestion: {congestion}"),
            f"Illegal parking IDs: {parking_ids}",
            f"Fire lane IDs: {fire_ids}",
            (
                f"Fire obstruction: {obstruction}"
                f"    Area: {snapshot.fire_lane_obstruction_area_ratio:.2%}"
            ),
        )
        for index, text in enumerate(lines):
            color = (245, 245, 245)
            if (
                "ALERT" in text
                or "SCENE CHANGE" in text
                or (index == 2 and not text.endswith("-"))
                or (index >= 4 and not text.endswith("-"))
            ):
                color = ALERT_COLOR
            cv2.putText(
                frame,
                text,
                (12, 27 + index * 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def track_color(
        track_id: int,
        snapshot: AnalyticsSnapshot,
        default: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        if track_id in snapshot.fire_lane_track_ids:
            return ALERT_COLOR
        if track_id in snapshot.visitor_loitering_track_ids:
            return ALERT_COLOR
        if track_id in snapshot.illegal_parking_track_ids:
            return PARKING_REGION_COLOR
        return default

    @staticmethod
    def _region_style(
        name: str,
        scene: SceneConfig,
        snapshot: AnalyticsSnapshot,
    ) -> tuple[tuple[int, int, int], bool]:
        loitering_regions = {
            zone.region for zone in scene.visitor_loitering.zones
        }
        if name in loitering_regions:
            return PEOPLE_REGION_COLOR, name in snapshot.visitor_loitering_regions
        if name == scene.fire_lane.region or (
            name == scene.fire_lane_obstruction.region
        ):
            return FIRE_LANE_REGION_COLOR, (
                bool(snapshot.fire_lane_track_ids)
                or snapshot.fire_lane_obstruction_active
            )
        if name == scene.illegal_parking.region:
            return PARKING_REGION_COLOR, bool(snapshot.illegal_parking_track_ids)
        if name == scene.congestion.region:
            return CONGESTION_REGION_COLOR, snapshot.congestion_active
        if name == scene.person_count.region:
            return PEOPLE_REGION_COLOR, False
        return DEFAULT_REGION_COLOR, False


def _format_ids(track_ids: tuple[int, ...]) -> str:
    if not track_ids:
        return "-"
    visible = ",".join(str(track_id) for track_id in track_ids[:6])
    return f"{visible},..." if len(track_ids) > 6 else visible

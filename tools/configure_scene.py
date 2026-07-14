from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


WINDOW_NAME = "Scene Configuration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create normalized analytics regions from the first video frame"
    )
    parser.add_argument(
        "--source", required=True, help="Local video path or camera index"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output scene YAML")
    parser.add_argument("--scene-id", default="scene_01")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame = read_first_frame(args.source)
    display = resize_for_display(frame)
    print("Left click to add points; Backspace undoes; Enter confirms; Esc cancels.")

    shapes = {
        "people_area": collect_points(display, "people_area", minimum=3),
        "congestion_area": collect_points(display, "congestion_area", minimum=3),
        "no_parking_area": collect_points(display, "no_parking_area", minimum=3),
        "fire_lane": collect_points(display, "fire_lane", minimum=3),
        "visitor_watch_area": collect_points(
            display, "visitor_watch_area", minimum=3
        ),
        "entrance": collect_points(display, "entrance", minimum=2, maximum=2),
    }
    cv2.destroyAllWindows()
    document = build_scene_document(args.scene_id, shapes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Scene configuration written to {args.output.resolve()}")
    print("If In/Out are reversed, swap entrance points or use positive_to_negative.")
    return 0


def read_first_frame(source: str) -> np.ndarray:
    normalized_source: str | int = int(source) if source.isdecimal() else source
    capture = cv2.VideoCapture(normalized_source)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read first frame: {source}")
        return frame
    finally:
        capture.release()


def resize_for_display(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 720 / height)
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def collect_points(
    frame: np.ndarray,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> list[list[float]]:
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, parameter) -> None:
        del flags, parameter
        if event == cv2.EVENT_LBUTTONDOWN and (
            maximum is None or len(points) < maximum
        ):
            points.append((x, y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(f"Configure {name}: add at least {minimum} point(s), then press Enter.")
    while True:
        canvas = frame.copy()
        for index, point in enumerate(points):
            cv2.circle(canvas, point, 5, (40, 220, 240), -1, cv2.LINE_AA)
            if index:
                cv2.line(
                    canvas,
                    points[index - 1],
                    point,
                    (40, 220, 240),
                    2,
                    cv2.LINE_AA,
                )
        if maximum is None and len(points) >= minimum:
            cv2.line(canvas, points[-1], points[0], (40, 220, 240), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) >= minimum:
            break
        if key in (8, 127) and points:
            points.pop()
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt

    height, width = frame.shape[:2]
    return [[round(x / width, 6), round(y / height, 6)] for x, y in points]


def build_scene_document(scene_id: str, shapes: dict[str, list[list[float]]]) -> dict:
    return {
        "scene": {"id": scene_id},
        "regions": {
            name: {"polygon": shapes[name]}
            for name in (
                "people_area",
                "congestion_area",
                "no_parking_area",
                "fire_lane",
                "visitor_watch_area",
            )
        },
        "lines": {
            "entrance": {
                "points": shapes["entrance"],
                "in_direction": "negative_to_positive",
                "hysteresis": 0.01,
                "cooldown_seconds": 1.0,
            }
        },
        "rules": {
            "person_count": {
                "enabled": True,
                "region": "people_area",
                "lines": ["entrance"],
            },
            "congestion": {
                "enabled": True,
                "region": "congestion_area",
                "min_vehicles": 4,
                "max_speed_ratio": 0.01,
                "min_low_speed_ratio": 0.7,
                "hold_seconds": 10,
                "recovery_seconds": 5,
            },
            "illegal_parking": {
                "enabled": True,
                "region": "no_parking_area",
                "max_speed_ratio": 0.005,
                "hold_seconds": 30,
                "recovery_seconds": 2,
                "suppress_when_congested": True,
            },
            "fire_lane": {
                "enabled": True,
                "region": "fire_lane",
                "max_speed_ratio": 0.005,
                "hold_seconds": 10,
                "recovery_seconds": 2,
            },
            "visitor_loitering": {
                "enabled": True,
                "include_roles": ["visitor", "unknown"],
                "period": "auto",
                "day_start": "06:00",
                "night_start": "18:00",
                "zones": [
                    {
                        "region": "visitor_watch_area",
                        "day_hold_seconds": 120,
                        "night_hold_seconds": 30,
                        "absence_grace_seconds": 3,
                        "recovery_seconds": 2,
                    }
                ],
            },
        },
        "analytics": {
            "history_seconds": 2,
            "stale_track_seconds": 2,
            "metrics_interval_seconds": 1,
        },
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Scene configuration cancelled.")
        raise SystemExit(130) from None

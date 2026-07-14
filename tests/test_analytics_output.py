from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analytics.schemas import AlertEvent, AnalysisResult, AnalyticsSnapshot
from src.config import OutputConfig
from src.result_writer import ResultWriter


def snapshot(frame_id: int, timestamp: float) -> AnalyticsSnapshot:
    return AnalyticsSnapshot(
        scene_id="test_scene",
        frame_id=frame_id,
        timestamp=timestamp,
        current_people=2,
        entries=3,
        exits=1,
        current_vehicles=1,
        region_vehicle_counts={"fire": 1},
        congestion_active=False,
        congestion_vehicle_count=1,
        mean_vehicle_speed_ratio=0.0,
        low_speed_vehicle_ratio=1.0,
        illegal_parking_track_ids=(),
        fire_lane_track_ids=(7,),
        active_event_types=("fire_lane_occupied",),
    )


class AnalyticsOutputTests(unittest.TestCase):
    def test_metrics_events_images_and_summary_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = OutputConfig(
                directory=root,
                save_video=False,
                save_jsonl=False,
            )
            event = AlertEvent(
                event_id="abc123456789",
                event_type="fire_lane_occupied",
                state="started",
                severity="critical",
                scene_id="test_scene",
                region="fire",
                timestamp=0.0,
                frame_id=0,
                track_ids=(7,),
            )
            frame = np.zeros((80, 120, 3), dtype=np.uint8)

            with ResultWriter(
                config,
                source_fps=10.0,
                analytics_enabled=True,
                metrics_interval_seconds=0.5,
            ) as writer:
                writer.write(frame, [], AnalysisResult(snapshot(0, 0.0), (event,)))
                writer.write(frame, [], AnalysisResult(snapshot(1, 0.2)))
                writer.write(frame, [], AnalysisResult(snapshot(2, 0.5)))

            metrics = (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            events = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            event_record = json.loads(events[0])

            self.assertEqual(len(metrics), 2)
            self.assertEqual(len(events), 1)
            self.assertEqual(summary["processed_frames"], 3)
            self.assertEqual(summary["event_counts"], {"fire_lane_occupied": 1})
            self.assertTrue((root / event_record["snapshot_path"]).is_file())


if __name__ == "__main__":
    unittest.main()

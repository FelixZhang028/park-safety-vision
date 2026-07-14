from __future__ import annotations

import unittest

from tools.evaluate import EventInterval, evaluate, temporal_iou


class EvaluateToolTests(unittest.TestCase):
    def test_event_metrics_and_count_errors(self) -> None:
        truth = {
            "counts": {"entries": 5, "exits": 2},
            "events": [
                {
                    "event_type": "illegal_parking",
                    "region": "parking",
                    "start": 10.0,
                    "end": 20.0,
                }
            ],
        }
        predictions = [
            EventInterval("illegal_parking", "parking", 11.0, 20.0),
            EventInterval("traffic_congestion", "traffic", 30.0, 40.0),
        ]
        summary = {"final_metrics": {"entries": 4, "exits": 2}}

        report = evaluate(
            truth,
            predictions,
            summary,
            min_temporal_iou=0.3,
            start_tolerance=2.0,
        )

        self.assertEqual(report["counts"]["entries"]["absolute_error"], 1)
        self.assertEqual(report["events"]["true_positive"], 1)
        self.assertEqual(report["events"]["false_positive"], 1)
        self.assertEqual(report["events"]["recall"], 1.0)
        self.assertEqual(report["events"]["mean_detection_delay_seconds"], 1.0)

    def test_temporal_iou(self) -> None:
        first = EventInterval("x", "r", 0.0, 10.0)
        second = EventInterval("x", "r", 5.0, 15.0)
        self.assertAlmostEqual(temporal_iou(first, second), 1 / 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EventInterval:
    event_type: str
    region: str
    start: float
    end: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare analytics events and counts with labeled ground truth"
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-temporal-iou", type=float, default=0.30)
    parser.add_argument("--start-tolerance", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    summary_path = args.summary or args.events.with_name("summary.json")
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    video_end = float((summary.get("final_metrics") or {}).get("timestamp", 0.0))
    predictions = load_prediction_intervals(args.events, video_end)
    report = evaluate(
        truth,
        predictions,
        summary,
        min_temporal_iou=args.min_temporal_iou,
        start_tolerance=args.start_tolerance,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


def load_prediction_intervals(path: Path, video_end: float) -> list[EventInterval]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    active: dict[str, EventInterval] = {}
    completed: list[EventInterval] = []
    for record in records:
        event_id = str(record["event_id"])
        if record["state"] == "started":
            active[event_id] = EventInterval(
                event_type=str(record["event_type"]),
                region=str(record["region"]),
                start=float(record["timestamp"]),
                end=video_end,
            )
        elif record["state"] == "ended" and event_id in active:
            interval = active.pop(event_id)
            interval.end = float(record["timestamp"])
            completed.append(interval)
    completed.extend(active.values())
    return completed


def evaluate(
    truth: dict[str, Any],
    predictions: list[EventInterval],
    summary: dict[str, Any],
    *,
    min_temporal_iou: float,
    start_tolerance: float,
) -> dict[str, Any]:
    expected = [
        EventInterval(
            event_type=str(item["event_type"]),
            region=str(item["region"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in truth.get("events", [])
    ]
    matches: list[tuple[int, int, float]] = []
    used_predictions: set[int] = set()
    for truth_index, expected_event in enumerate(expected):
        candidates: list[tuple[float, int]] = []
        for prediction_index, predicted_event in enumerate(predictions):
            if prediction_index in used_predictions:
                continue
            if (
                predicted_event.event_type != expected_event.event_type
                or predicted_event.region != expected_event.region
            ):
                continue
            iou = temporal_iou(expected_event, predicted_event)
            if (
                iou >= min_temporal_iou
                or abs(predicted_event.start - expected_event.start) <= start_tolerance
            ):
                candidates.append((iou, prediction_index))
        if candidates:
            _, prediction_index = max(candidates)
            used_predictions.add(prediction_index)
            delay = predictions[prediction_index].start - expected_event.start
            matches.append((truth_index, prediction_index, delay))

    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(expected) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    delays = [match[2] for match in matches]

    final_metrics = summary.get("final_metrics") or {}
    count_truth = truth.get("counts", {})
    count_report = {
        name: {
            "expected": int(count_truth[name]),
            "predicted": int(final_metrics.get(name, 0)),
            "absolute_error": abs(
                int(final_metrics.get(name, 0)) - int(count_truth[name])
            ),
        }
        for name in ("entries", "exits", "current_people", "current_vehicles")
        if name in count_truth
    }
    duration_seconds = float(final_metrics.get("timestamp", 0.0))
    false_alarms_per_hour = _ratio(false_positive * 3600, duration_seconds)
    return {
        "counts": count_report,
        "events": {
            "expected": len(expected),
            "predicted": len(predictions),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "mean_detection_delay_seconds": (
                round(sum(delays) / len(delays), 6) if delays else None
            ),
            "false_alarms_per_hour": (
                round(false_alarms_per_hour, 6) if duration_seconds > 0 else None
            ),
        },
    }


def temporal_iou(first: EventInterval, second: EventInterval) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return _ratio(intersection, union)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

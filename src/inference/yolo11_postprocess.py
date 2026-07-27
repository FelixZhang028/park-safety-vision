from __future__ import annotations

import numpy as np

from .preprocessing import LetterboxInfo
from .schemas import COCO_CLASS_NAMES, DetectionResult


def decode_yolo11_output(
    output: np.ndarray,
    letterbox: LetterboxInfo,
    confidence_threshold: float,
    iou_threshold: float,
    allowed_classes: tuple[int, ...],
) -> list[DetectionResult]:
    predictions = _prediction_rows(output)
    if predictions.shape[1] < 5:
        raise ValueError(f"Unexpected YOLO output shape: {np.asarray(output).shape}")

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(predictions)), class_ids]
    allowed = np.isin(class_ids, np.asarray(allowed_classes, dtype=np.int64))
    keep = (confidences >= confidence_threshold) & allowed
    if not np.any(keep):
        return []

    boxes = _xywh_to_xyxy(predictions[keep, :4])
    class_ids = class_ids[keep].astype(np.int64)
    confidences = confidences[keep].astype(np.float32)
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        class_keep = _nms(boxes[indices], confidences[indices], iou_threshold)
        selected.extend(int(indices[index]) for index in class_keep)
    selected.sort(key=lambda index: float(confidences[index]), reverse=True)

    detections: list[DetectionResult] = []
    for index in selected:
        restored = _restore_box(boxes[index], letterbox)
        if restored[2] <= restored[0] or restored[3] <= restored[1]:
            continue
        class_id = int(class_ids[index])
        detections.append(
            DetectionResult(
                class_id=class_id,
                class_name=COCO_CLASS_NAMES.get(class_id, str(class_id)),
                confidence=float(confidences[index]),
                bbox=restored,
            )
        )
    return detections


def _prediction_rows(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {np.asarray(output).shape}")
    if array.shape[0] in {84, 85} or (
        array.shape[0] < array.shape[1] and array.shape[0] <= 256
    ):
        array = array.T
    return array


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _restore_box(
    box: np.ndarray, info: LetterboxInfo
) -> tuple[float, float, float, float]:
    x1 = np.clip((box[0] - info.pad_x) / info.scale, 0, info.original_width)
    y1 = np.clip((box[1] - info.pad_y) / info.scale, 0, info.original_height)
    x2 = np.clip((box[2] - info.pad_x) / info.scale, 0, info.original_width)
    y2 = np.clip((box[3] - info.pad_y) / info.scale, 0, info.original_height)
    return float(x1), float(y1), float(x2), float(y2)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = _iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[overlaps <= threshold]
    return keep


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )

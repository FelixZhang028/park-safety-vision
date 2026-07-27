from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def prepare_rknn_input(
    frame: np.ndarray, image_size: int
) -> tuple[np.ndarray, LetterboxInfo]:
    height, width = frame.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    pad_x = (image_size - resized_width) / 2.0
    pad_y = (image_size - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    batched = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)
    return batched, LetterboxInfo(scale, float(left), float(top), width, height)

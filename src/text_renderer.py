from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(slots=True, frozen=True)
class TextItem:
    text: str
    position: tuple[int, int]
    size: int
    color: tuple[int, int, int]
    bold: bool = False


class UnicodeTextRenderer:
    """Draw Chinese presentation labels while keeping OpenCV frames in BGR."""

    def __init__(self) -> None:
        self.regular_path = _find_font(bold=False)
        self.bold_path = _find_font(bold=True) or self.regular_path

    def draw(self, frame: np.ndarray, items: Sequence[TextItem]) -> None:
        if not items:
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for item in items:
            draw.text(
                item.position,
                item.text,
                font=self._font(item.size, item.bold),
                fill=_bgr_to_rgb(item.color),
            )
        frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def measure(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        left, top, right, bottom = self._font(size, bold).getbbox(text)
        return right - left, bottom - top

    @lru_cache(maxsize=32)
    def _font(self, size: int, bold: bool) -> ImageFont.FreeTypeFont:
        path = self.bold_path if bold else self.regular_path
        if path is not None:
            return ImageFont.truetype(str(path), size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)


def _find_font(bold: bool) -> Path | None:
    override = os.environ.get(
        "PARK_SAFETY_BOLD_FONT" if bold else "PARK_SAFETY_FONT"
    )
    candidates = [Path(override)] if override else []
    if bold:
        candidates.extend(
            (
                Path("C:/Windows/Fonts/msyhbd.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            )
        )
    else:
        candidates.extend(
            (
                Path("C:/Windows/Fonts/msyh.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            )
        )
    return next((path for path in candidates if path.is_file()), None)


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]

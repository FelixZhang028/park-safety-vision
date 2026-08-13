from __future__ import annotations

import unittest

import numpy as np

from src.text_renderer import TextItem, UnicodeTextRenderer


class UnicodeTextRendererTests(unittest.TestCase):
    def test_chinese_text_is_measured_and_drawn(self) -> None:
        renderer = UnicodeTextRenderer()
        frame = np.zeros((80, 240, 3), dtype=np.uint8)

        width, height = renderer.measure("拥堵报警", 24, bold=True)
        renderer.draw(
            frame,
            [TextItem("拥堵报警", (10, 10), 24, (255, 255, 255), True)],
        )

        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertGreater(np.count_nonzero(frame), 0)


if __name__ == "__main__":
    unittest.main()

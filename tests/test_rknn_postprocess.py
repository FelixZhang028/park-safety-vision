from __future__ import annotations

import unittest

import numpy as np

from src.inference.preprocessing import LetterboxInfo, prepare_rknn_input
from src.inference.yolo11_postprocess import decode_yolo11_output


class RknnPrePostProcessingTests(unittest.TestCase):
    def test_input_is_uint8_nhwc_with_letterbox(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tensor, info = prepare_rknn_input(frame, 640)

        self.assertEqual(tensor.shape, (1, 640, 640, 3))
        self.assertEqual(tensor.dtype, np.uint8)
        self.assertAlmostEqual(info.scale, 0.5)
        self.assertEqual(info.pad_x, 0.0)
        self.assertEqual(info.pad_y, 140.0)

    def test_output_is_filtered_nms_applied_and_coordinates_restored(self) -> None:
        output = np.zeros((1, 84, 3), dtype=np.float32)
        output[0, :4, 0] = [320.0, 320.0, 100.0, 200.0]
        output[0, 4, 0] = 0.90
        output[0, :4, 1] = [322.0, 320.0, 100.0, 200.0]
        output[0, 4, 1] = 0.80
        output[0, :4, 2] = [100.0, 100.0, 50.0, 50.0]
        output[0, 5, 2] = 0.99
        info = LetterboxInfo(1.0, 0.0, 0.0, 640, 640)

        detections = decode_yolo11_output(
            output,
            letterbox=info,
            confidence_threshold=0.35,
            iou_threshold=0.50,
            allowed_classes=(0, 2),
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(detections[0].bbox, (270.0, 220.0, 370.0, 420.0))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppConfigTests(unittest.TestCase):
    def test_default_project_config_loads(self) -> None:
        config = AppConfig.from_yaml(PROJECT_ROOT / "config.yaml")

        self.assertEqual(config.model.path, "yolo11n.pt")
        self.assertEqual(config.model.classes, (0, 2, 3, 5, 7))
        self.assertEqual(config.output.directory, PROJECT_ROOT / "outputs")
        self.assertEqual(config.tracking.tracker, "bytetrack.yaml")

    def test_relative_paths_are_resolved_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "custom.yaml"
            config_path.write_text(
                "model:\n  path: models/model.pt\n"
                "tracking:\n  tracker: trackers/custom.yaml\n"
                "output:\n  directory: results\n",
                encoding="utf-8",
            )

            config = AppConfig.from_yaml(config_path)

            self.assertEqual(config.model.path, str(root / "models" / "model.pt"))
            self.assertEqual(
                config.tracking.tracker, str(root / "trackers" / "custom.yaml")
            )
            self.assertEqual(config.output.directory, root / "results")

    def test_rk3588_config_loads(self) -> None:
        config = AppConfig.from_yaml(PROJECT_ROOT / "config.rk3588.yaml")

        self.assertEqual(config.model.backend, "rknn")
        self.assertEqual(config.model.npu_core, "auto")
        self.assertTrue(config.model.path.endswith("models\\yolo11n_fp16.rknn"))
        self.assertFalse(config.display.show_window)

    def test_demo_config_enables_clean_presentation_output(self) -> None:
        config = AppConfig.from_yaml(PROJECT_ROOT / "config.demo.yaml")

        self.assertTrue(config.display.presentation_mode)
        self.assertFalse(config.display.show_track_labels)
        self.assertFalse(config.display.show_fps)
        self.assertEqual(config.display.output_width, 1280)
        self.assertEqual(config.display.output_height, 720)

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid.yaml"
            config_path.write_text("model:\n  confidnce: 0.5\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "confidnce"):
                AppConfig.from_yaml(config_path)

    def test_invalid_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid.yaml"
            config_path.write_text("model:\n  confidence: 1.5\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "confidence"):
                AppConfig.from_yaml(config_path)


if __name__ == "__main__":
    unittest.main()

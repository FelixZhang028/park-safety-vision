from __future__ import annotations

import unittest
from pathlib import Path

from src.config import ConfigurationError
from src.scene_config import SceneConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SceneConfigTests(unittest.TestCase):
    def test_example_scene_loads(self) -> None:
        scene = SceneConfig.from_yaml(PROJECT_ROOT / "scenes" / "example.yaml")

        self.assertEqual(scene.scene_id, "offline_demo")
        self.assertTrue(scene.person_count.enabled)
        self.assertEqual(scene.person_count.lines, ("entrance",))
        self.assertEqual(scene.congestion.min_vehicles, 4)
        self.assertIn("fire_lane", scene.regions)
        self.assertTrue(scene.visitor_loitering.enabled)
        self.assertEqual(
            scene.visitor_loitering.zones[0].region,
            "visitor_watch_area",
        )

    def test_obstruction_scene_resolves_baseline_relative_to_scene(self) -> None:
        scene = SceneConfig.from_yaml(
            PROJECT_ROOT / "scenes" / "aboda_obstruction.yaml"
        )

        self.assertTrue(scene.fire_lane_obstruction.enabled)
        self.assertEqual(
            scene.fire_lane_obstruction.baseline_path,
            (PROJECT_ROOT / "data" / "baselines" / "aboda_video1.jpg").resolve(),
        )

    def test_demo_scenes_load_with_presentation_metadata(self) -> None:
        people = SceneConfig.from_yaml(
            PROJECT_ROOT / "scenes" / "demo" / "people_count.yaml"
        )
        congestion = SceneConfig.from_yaml(
            PROJECT_ROOT / "scenes" / "demo" / "gate_congestion.yaml"
        )
        clutter = SceneConfig.from_yaml(
            PROJECT_ROOT / "scenes" / "demo" / "public_area_clutter.yaml"
        )

        self.assertEqual(people.display_name, "人员进出统计")
        self.assertTrue(people.person_count.enabled)
        self.assertTrue(congestion.congestion.enabled)
        self.assertEqual(congestion.congestion.min_vehicles, 20)
        self.assertTrue(clutter.fire_lane_obstruction.enabled)
        self.assertTrue(clutter.fire_lane_obstruction.baseline_path.is_file())

    def test_unknown_region_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "unknown region"):
            SceneConfig.from_mapping(
                {
                    "scene": {"id": "invalid"},
                    "rules": {
                        "congestion": {
                            "enabled": True,
                            "region": "missing",
                        }
                    },
                }
            )

    def test_non_normalized_polygon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "normalized"):
            SceneConfig.from_mapping(
                {
                    "regions": {
                        "bad": {
                            "polygon": [[0, 0], [2, 0], [0, 1]],
                        }
                    }
                }
            )

    def test_unknown_loitering_region_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "unknown region"):
            SceneConfig.from_mapping(
                {
                    "rules": {
                        "visitor_loitering": {
                            "enabled": True,
                            "zones": [{"region": "missing"}],
                        }
                    }
                }
            )

    def test_invalid_loitering_clock_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Invalid clock time"):
            SceneConfig.from_mapping(
                {
                    "rules": {
                        "visitor_loitering": {
                            "enabled": False,
                            "day_start": "25:00",
                        }
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()

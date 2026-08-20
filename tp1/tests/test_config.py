import json
import tempfile
import unittest
from pathlib import Path

from sia_tp1.config import ConfigError, load_config


class ConfigTest(unittest.TestCase):
    def test_load_config_and_resolve_relative_level_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            self._write_config(config_path)

            config = load_config(config_path)

            self.assertEqual(config.level_file, root / "levels/level.txt")
            self.assertEqual(config.algorithm, "bfs")
            self.assertIsNone(config.heuristic)
            self.assertEqual(config.cost_model, "unit")
            self.assertIsNone(config.limits.max_expanded_nodes)
            self.assertIsNone(config.limits.timeout_seconds)
            self.assertEqual(config.seed, 0)

    def test_reject_unknown_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_config(config_path, algorithm="unknown")

            with self.assertRaisesRegex(ConfigError, "algorithm must be"):
                load_config(config_path)

    def test_reject_non_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_config(config_path, max_expanded_nodes=0)

            with self.assertRaisesRegex(ConfigError, "positive integer"):
                load_config(config_path)

    def test_reject_heuristic_for_uninformed_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_config(
                config_path,
                heuristic="minimum_matching_manhattan",
            )

            with self.assertRaisesRegex(ConfigError, "requires heuristic.*null"):
                load_config(config_path)

    def test_require_heuristic_for_informed_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_config(config_path, algorithm="greedy")

            with self.assertRaisesRegex(ConfigError, "requires a heuristic"):
                load_config(config_path)

    def test_reject_unknown_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_config(
                config_path,
                algorithm="astar",
                heuristic="unknown",
            )

            with self.assertRaisesRegex(ConfigError, "Unknown heuristic"):
                load_config(config_path)

    def test_accept_registered_heuristic_for_informed_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_config(
                config_path,
                algorithm="astar",
                heuristic="shortest_push_access",
            )

            config = load_config(config_path)

            self.assertEqual(config.algorithm, "astar")
            self.assertEqual(config.heuristic, "shortest_push_access")

    def _write_config(
        self,
        path: Path,
        algorithm: str = "bfs",
        heuristic=None,
        max_expanded_nodes=None,
    ) -> None:
        data = {
            "level_file": "levels/level.txt",
            "algorithm": algorithm,
            "heuristic": heuristic,
            "cost_model": "unit",
            "limits": {
                "max_expanded_nodes": max_expanded_nodes,
                "timeout_seconds": None,
            },
            "seed": 0,
        }
        path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

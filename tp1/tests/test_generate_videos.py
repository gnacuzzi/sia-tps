import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.generate_videos import CASES, generate_all_videos


class GenerateVideosScriptTest(unittest.TestCase):
    def test_runs_six_cases_and_restores_exact_original_config(self) -> None:
        observed_cases = []

        with tempfile.TemporaryDirectory() as directory:
            temporary_directory = Path(directory)
            config_path = temporary_directory / "config.json"
            original_config = b'{"custom_spacing": true, "seed": 7}\n'
            config_path.write_bytes(original_config)
            level_path = Path("levels/level_01.txt").resolve()
            output_dir = temporary_directory / "videos"

            def fake_runner(command, **kwargs):
                current = json.loads(config_path.read_text(encoding="utf-8"))
                observed_cases.append(
                    (current["algorithm"], current["heuristic"], command[-1])
                )
                self.assertEqual(kwargs["check"], False)
                self.assertIn("PYTHONPATH", kwargs["env"])
                self.assertIn("--video", command)
                self.assertIn("--video-max-seconds", command)
                return subprocess.CompletedProcess(command, 0)

            exit_code = generate_all_videos(
                level_path,
                config_path,
                output_dir,
                runner=fake_runner,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(config_path.read_bytes(), original_config)

        self.assertEqual(
            [(algorithm, heuristic) for algorithm, heuristic, _ in observed_cases],
            [(algorithm, heuristic) for algorithm, heuristic, _ in CASES],
        )
        self.assertEqual(
            [Path(path).name for _, _, path in observed_cases],
            [filename for _, _, filename in CASES],
        )

    def test_restores_config_when_one_case_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_directory = Path(directory)
            config_path = temporary_directory / "config.json"
            original_config = b'{"seed": 3}\n'
            config_path.write_bytes(original_config)

            def failing_runner(command, **kwargs):
                return subprocess.CompletedProcess(command, 9)

            exit_code = generate_all_videos(
                Path("levels/level_01.txt").resolve(),
                config_path,
                temporary_directory / "videos",
                runner=failing_runner,
            )

            self.assertEqual(exit_code, 9)
            self.assertEqual(config_path.read_bytes(), original_config)


if __name__ == "__main__":
    unittest.main()

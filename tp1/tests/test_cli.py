import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sia_tp1.cli import main, run, run_search
from sia_tp1.search import a_star_search, depth_first_search, greedy_search


class CliTest(unittest.TestCase):
    def test_reject_gif_without_search(self) -> None:
        error_output = io.StringIO()

        with patch("sys.stderr", error_output):
            exit_code = main(["--gif", "solution.gif"])

        self.assertEqual(exit_code, 2)
        self.assertIn("--gif requires --search", error_output.getvalue())

    def test_reject_video_without_search(self) -> None:
        error_output = io.StringIO()

        with patch("sys.stderr", error_output):
            exit_code = main(["--video", "solution.mp4"])

        self.assertEqual(exit_code, 2)
        self.assertIn("--video requires --search", error_output.getvalue())

    def test_replay_manual_acceptance_sequence(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(Path(directory))
            exit_code = run(
                config_path,
                ["RIGHT", "RIGHT", "RIGHT"],
                output=output,
            )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Move 1: RIGHT, pushed=false", rendered)
        self.assertIn("Move 2: RIGHT, pushed=true", rendered)
        self.assertIn("Move 3: RIGHT, pushed=true", rendered)
        self.assertIn("#___@*#", rendered)
        self.assertIn("Solved: yes", rendered)
        self.assertIn("Valid moves: 3", rendered)
        self.assertIn("Pushes: 2", rendered)
        self.assertIn("Cost: 3", rendered)

    def test_invalid_move_does_not_add_cost(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(Path(directory))
            run(config_path, ["LEFT"], output=output)

        rendered = output.getvalue()
        self.assertIn("Move 1: LEFT, invalid", rendered)
        self.assertIn("Valid moves: 0", rendered)
        self.assertIn("Cost: 0", rendered)

    def test_run_configured_bfs(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(Path(directory))
            exit_code = run_search(config_path, output=output)

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Status: success", rendered)
        self.assertIn("Expanded nodes: 4", rendered)
        self.assertIn("Frontier at end: 1", rendered)
        self.assertIn("Maximum frontier: 2", rendered)
        self.assertIn("Solution cost: 3", rendered)
        self.assertIn("Solution moves: 3", rendered)
        self.assertIn("Solution pushes: 2", rendered)
        self.assertIn("3: RIGHT, pushed=true", rendered)
        self.assertIn("#___@*#", rendered)

    def test_run_search_writes_gif_to_nested_directory(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(Path(directory))
            gif_path = Path(directory) / "output" / "videos" / "bfs.gif"

            exit_code = run_search(
                config_path,
                output=output,
                gif_path=gif_path,
            )

            self.assertTrue(gif_path.is_file())

        self.assertEqual(exit_code, 0)
        self.assertIn("GIF saved to:", output.getvalue())
        self.assertIn("(4 frames)", output.getvalue())

    def test_run_search_writes_video_to_nested_directory(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(Path(directory))
            video_path = Path(directory) / "output" / "videos" / "bfs.mp4"

            with patch(
                "sia_tp1.cli.save_solution_video",
                return_value=4,
            ) as save_video:
                exit_code = run_search(
                    config_path,
                    output=output,
                    video_path=video_path,
                )

        self.assertEqual(exit_code, 0)
        save_video.assert_called_once()
        self.assertIn("Video saved to:", output.getvalue())
        self.assertIn("(4 frames)", output.getvalue())

    def test_video_duration_cap_only_reduces_frame_duration(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(Path(directory))
            video_path = Path(directory) / "bfs.mp4"

            with patch(
                "sia_tp1.cli.save_solution_video",
                return_value=4,
            ) as save_video:
                run_search(
                    config_path,
                    output=output,
                    video_path=video_path,
                    video_max_seconds=1,
                )

        self.assertEqual(save_video.call_args.kwargs["frame_duration_ms"], 250)
        self.assertIn("250 ms/frame", output.getvalue())

    def test_run_configured_dfs(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(
                Path(directory),
                algorithm="dfs",
                heuristic=None,
            )

            with patch(
                "sia_tp1.cli.depth_first_search",
                wraps=depth_first_search,
            ) as configured_dfs:
                exit_code = run_search(config_path, output=output)

        rendered = output.getvalue()
        configured_dfs.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertIn("Status: success", rendered)
        self.assertIn("Solution cost: 3", rendered)
        self.assertIn("Solution moves: 3", rendered)
        self.assertIn("Solution pushes: 2", rendered)
        self.assertIn("#___@*#", rendered)

    def test_run_configured_greedy(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(
                Path(directory),
                algorithm="greedy",
                heuristic="minimum_matching_manhattan",
            )

            with patch(
                "sia_tp1.cli.greedy_search",
                wraps=greedy_search,
            ) as configured_greedy:
                exit_code = run_search(config_path, output=output)

        rendered = output.getvalue()
        configured_greedy.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertIn("Status: success", rendered)
        self.assertIn("Solution cost: 3", rendered)
        self.assertIn("Solution moves: 3", rendered)
        self.assertIn("Solution pushes: 2", rendered)
        self.assertIn("#___@*#", rendered)

    def test_run_configured_astar(self) -> None:
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_search_config(
                Path(directory),
                algorithm="astar",
                heuristic="shortest_push_access",
            )

            with patch(
                "sia_tp1.cli.a_star_search",
                wraps=a_star_search,
            ) as configured_astar:
                exit_code = run_search(config_path, output=output)

        rendered = output.getvalue()
        configured_astar.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertIn("Status: success", rendered)
        self.assertIn("Solution cost: 3", rendered)
        self.assertIn("Solution moves: 3", rendered)
        self.assertIn("Solution pushes: 2", rendered)
        self.assertIn("#___@*#", rendered)

    def _write_search_config(
        self,
        directory: Path,
        *,
        algorithm: str = "bfs",
        heuristic=None,
    ) -> Path:
        config_path = directory / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "level_file": str(Path("levels/level_01.txt").resolve()),
                    "algorithm": algorithm,
                    "heuristic": heuristic,
                    "cost_model": "unit",
                    "limits": {
                        "max_expanded_nodes": None,
                        "timeout_seconds": None,
                    },
                    "seed": 0,
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()

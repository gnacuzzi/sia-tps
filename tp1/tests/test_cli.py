import io
import unittest
from pathlib import Path

from sia_tp1.cli import run, run_search


class CliTest(unittest.TestCase):
    def test_replay_manual_acceptance_sequence(self) -> None:
        output = io.StringIO()

        exit_code = run(
            Path("config.json"),
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

        run(Path("config.json"), ["LEFT"], output=output)

        rendered = output.getvalue()
        self.assertIn("Move 1: LEFT, invalid", rendered)
        self.assertIn("Valid moves: 0", rendered)
        self.assertIn("Cost: 0", rendered)

    def test_run_configured_bfs(self) -> None:
        output = io.StringIO()

        exit_code = run_search(Path("config.json"), output=output)

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


if __name__ == "__main__":
    unittest.main()

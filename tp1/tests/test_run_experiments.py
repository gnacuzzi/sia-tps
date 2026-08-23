import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_experiments import (
    CORE_CASES,
    EXTENSION_CASES,
    SUMMARY_FIELDS,
    collect_experiment_rows,
    write_summary_csv,
)
from sia_tp1.search import SearchLimits


class ExperimentRunnerTest(unittest.TestCase):
    def test_core_and_extension_suites_have_expected_cases(self) -> None:
        self.assertEqual(len(CORE_CASES), 6)
        self.assertEqual(len(EXTENSION_CASES), 4)
        self.assertEqual(len(set(CORE_CASES + EXTENSION_CASES)), 10)

    def test_collects_repetitions_and_aggregates_summary(self) -> None:
        calls = []

        def fake_runner(level_path, algorithm, heuristic, limits):
            calls.append((level_path, algorithm, heuristic, limits))
            return {
                "status": "success",
                "cutoff_reason": "",
                "solution_cost": 3.0,
                "solution_moves": 3,
                "solution_pushes": 2,
                "expanded_nodes": 4,
                "frontier_size_at_end": 1,
                "max_frontier_size": 2,
                "elapsed_seconds": 0.25,
                "solution": "RIGHT RIGHT RIGHT",
            }

        limits = SearchLimits(100, 2.0)
        rows = collect_experiment_rows(
            level_paths=[Path("levels/level_01.txt").resolve()],
            cases=CORE_CASES,
            repetitions=2,
            limits=limits,
            seed=7,
            case_runner=fake_runner,
        )

        self.assertEqual(len(rows), 12)
        self.assertEqual(len(calls), 12)
        self.assertEqual({row["repetition"] for row in rows}, {1, 2})
        self.assertEqual({row["max_expanded_nodes"] for row in rows}, {100})

        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.csv"
            write_summary_csv(rows, summary_path)
            with summary_path.open(encoding="utf-8", newline="") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))

        self.assertEqual(len(summary_rows), 6)
        self.assertEqual(tuple(summary_rows[0]), SUMMARY_FIELDS)
        self.assertEqual({row["runs"] for row in summary_rows}, {"2"})
        self.assertEqual({row["successes"] for row in summary_rows}, {"2"})
        self.assertEqual(
            {row["elapsed_seconds_median"] for row in summary_rows},
            {"0.25"},
        )


if __name__ == "__main__":
    unittest.main()

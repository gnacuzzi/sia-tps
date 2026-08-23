import csv
import tempfile
import unittest
from pathlib import Path

from scripts.generate_plots import generate_charts, method_label, read_raw_results
from scripts.run_experiments import RAW_FIELDS


class GeneratePlotsTest(unittest.TestCase):
    def test_reads_repetitions_and_generates_all_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "experiments.csv"
            row = {field: "" for field in RAW_FIELDS}
            row.update(
                {
                    "level": "levels/level_01.txt",
                    "algorithm": "astar",
                    "heuristic": "minimum_matching_manhattan",
                    "repetition": "1",
                    "status": "success",
                    "solution_cost": "3",
                    "expanded_nodes": "12",
                    "max_frontier_size": "5",
                    "elapsed_seconds": "0.02",
                }
            )
            with results.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=RAW_FIELDS)
                writer.writeheader()
                writer.writerow(row)

            rows = read_raw_results(results)
            generated = generate_charts(rows, root / "plots", dpi=72)

            self.assertEqual(len(generated), 5)
            self.assertTrue(all(path.is_file() for path in generated))
            self.assertEqual(method_label(("astar", "minimum_matching_manhattan")), "A* · MMM")

    def test_rejects_a_raw_experiment_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text("level,algorithm\nlevel.txt,bfs\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing columns"):
                read_raw_results(path)


if __name__ == "__main__":
    unittest.main()

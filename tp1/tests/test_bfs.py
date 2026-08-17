import tempfile
import unittest
from pathlib import Path

from sia_tp1.parser import parse_level
from sia_tp1.search import (
    CutoffReason,
    SearchLimits,
    SearchStatus,
    breadth_first_search,
)


class BreadthFirstSearchTest(unittest.TestCase):
    def test_solve_minimal_level_with_shortest_path(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        result = breadth_first_search(level, initial)

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.solution_cost, 3)
        self.assertEqual(result.solution_moves, 3)
        self.assertEqual(result.solution_pushes, 2)
        self.assertEqual(
            [transition.direction.name for transition in result.solution_transitions],
            ["RIGHT", "RIGHT", "RIGHT"],
        )
        self.assertEqual(result.expanded_nodes, 4)
        self.assertEqual(result.frontier_size_at_end, 1)
        self.assertEqual(result.max_frontier_size, 2)

    def test_initial_goal_is_success_without_expansion(self) -> None:
        level, initial = self._parse_text("#####\n#@*##\n#####\n")

        result = breadth_first_search(level, initial)

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 0)
        self.assertEqual(result.max_frontier_size, 1)
        self.assertEqual(result.solution_cost, 0)
        self.assertEqual(result.solution_moves, 0)

    def test_failure_when_frontier_is_exhausted(self) -> None:
        level, initial = self._parse_text("######\n#$@._#\n######\n")

        result = breadth_first_search(level, initial)

        self.assertEqual(result.status, SearchStatus.FAILURE)
        self.assertGreater(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 0)
        self.assertIsNone(result.solution_cost)

    def test_cutoff_before_expansion_beyond_configured_limit(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        result = breadth_first_search(
            level,
            initial,
            SearchLimits(max_expanded_nodes=1),
        )

        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(
            result.cutoff_reason, CutoffReason.MAX_EXPANDED_NODES
        )
        self.assertEqual(result.expanded_nodes, 1)
        self.assertEqual(result.frontier_size_at_end, 1)
        self.assertIsNone(result.solution_cost)

    def test_timeout_before_first_expansion(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        times = iter([10.0, 12.0, 12.0])

        result = breadth_first_search(
            level,
            initial,
            SearchLimits(timeout_seconds=1.0),
            clock=lambda: next(times),
        )

        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(result.cutoff_reason, CutoffReason.TIMEOUT)
        self.assertEqual(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 1)

    def _parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level.txt"
            path.write_text(text, encoding="utf-8")
            return parse_level(path)


if __name__ == "__main__":
    unittest.main()


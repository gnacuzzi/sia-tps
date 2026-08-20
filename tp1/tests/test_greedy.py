import tempfile
import unittest
from pathlib import Path

from sia_tp1.heuristics import minimum_matching_manhattan_distance
from sia_tp1.parser import parse_level
from sia_tp1.search import (
    CutoffReason,
    SearchLimits,
    SearchStatus,
    greedy_search,
)


class GreedySearchTest(unittest.TestCase):
    def test_solves_minimal_level_with_student_heuristic(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        result = greedy_search(
            level,
            initial,
            minimum_matching_manhattan_distance,
        )

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.solution_cost, 3)
        self.assertEqual(result.solution_moves, 3)
        self.assertEqual(result.solution_pushes, 2)
        self.assertEqual(
            [
                transition.direction.name
                for transition in result.solution_transitions
            ],
            ["RIGHT", "RIGHT", "RIGHT"],
        )

    def test_initial_goal_is_success_without_expansion(self) -> None:
        level, initial = self._parse_text("#####\n#@*##\n#####\n")

        result = greedy_search(level, initial, lambda _level, _state: 0)

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 0)
        self.assertEqual(result.max_frontier_size, 1)
        self.assertEqual(result.solution_cost, 0)

    def test_insertion_order_breaks_equal_priority_ties(self) -> None:
        level, initial = self._parse_text(
            "#####\n#_.##\n#_$##\n#_@_#\n#####\n"
        )

        result = greedy_search(level, initial, lambda _level, _state: 0)

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.expanded_nodes, 1)
        self.assertEqual(
            [
                transition.direction.name
                for transition in result.solution_transitions
            ],
            ["UP"],
        )

    def test_failure_when_frontier_is_exhausted(self) -> None:
        level, initial = self._parse_text("######\n#$@._#\n######\n")

        result = greedy_search(level, initial, lambda _level, _state: 0)

        self.assertEqual(result.status, SearchStatus.FAILURE)
        self.assertGreater(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 0)
        self.assertIsNone(result.solution_cost)

    def test_cutoff_before_expansion_beyond_configured_limit(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        result = greedy_search(
            level,
            initial,
            minimum_matching_manhattan_distance,
            SearchLimits(max_expanded_nodes=1),
        )

        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(
            result.cutoff_reason, CutoffReason.MAX_EXPANDED_NODES
        )
        self.assertEqual(result.expanded_nodes, 1)
        self.assertEqual(result.frontier_size_at_end, 1)

    def test_timeout_before_first_expansion(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        times = iter([10.0, 12.0, 12.0])

        result = greedy_search(
            level,
            initial,
            minimum_matching_manhattan_distance,
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

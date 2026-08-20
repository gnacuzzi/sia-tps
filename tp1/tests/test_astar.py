import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sia_tp1.heuristics import minimum_matching_manhattan_distance
from sia_tp1.model import Direction, Level, State, Transition
from sia_tp1.parser import parse_level
from sia_tp1.search import (
    CutoffReason,
    SearchLimits,
    SearchStatus,
    a_star_search,
    breadth_first_search,
)


class AStarSearchTest(unittest.TestCase):
    def test_matches_bfs_optimal_cost_on_minimal_level(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        bfs_result = breadth_first_search(level, initial)
        astar_result = a_star_search(
            level,
            initial,
            minimum_matching_manhattan_distance,
        )

        self.assertEqual(astar_result.status, SearchStatus.SUCCESS)
        self.assertEqual(astar_result.solution_cost, bfs_result.solution_cost)
        self.assertEqual(astar_result.solution_cost, 3)
        self.assertEqual(astar_result.solution_moves, 3)
        self.assertEqual(astar_result.solution_pushes, 2)

    def test_initial_goal_is_success_without_expansion(self) -> None:
        level, initial = self._parse_text("#####\n#@*##\n#####\n")

        result = a_star_search(level, initial, lambda _level, _state: 0)

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 0)
        self.assertEqual(result.max_frontier_size, 1)
        self.assertEqual(result.solution_cost, 0)

    def test_reopens_a_state_when_a_lower_g_is_found(self) -> None:
        states = {
            name: State(player=(0, index), boxes=frozenset())
            for index, name in enumerate(("S", "A", "B", "C", "X", "G"))
        }
        edges = {
            (states["S"], Direction.UP): states["A"],
            (states["S"], Direction.DOWN): states["B"],
            (states["A"], Direction.UP): states["C"],
            (states["C"], Direction.UP): states["X"],
            (states["B"], Direction.UP): states["X"],
            (states["X"], Direction.UP): states["G"],
        }
        heuristic_values = {
            states["S"]: 0,
            states["A"]: 0,
            states["B"]: 10,
            states["C"]: 0,
            states["X"]: 0,
            states["G"]: 100,
        }

        def fake_apply_move(_level, state, direction):
            child_state = edges.get((state, direction))
            if child_state is None:
                return None
            return Transition(child_state, direction, pushed=False)

        level = Level(
            walls=frozenset(),
            floors=frozenset({state.player for state in states.values()}),
            goals=frozenset(),
            height=1,
            width=6,
        )

        with patch("sia_tp1.search.astar.apply_move", fake_apply_move), patch(
            "sia_tp1.search.astar.is_goal",
            lambda _level, state: state == states["G"],
        ):
            result = a_star_search(
                level,
                states["S"],
                lambda _level, state: heuristic_values[state],
            )

        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.solution_cost, 3)
        self.assertEqual(result.frontier_size_at_end, 0)

    def test_failure_when_frontier_is_exhausted(self) -> None:
        level, initial = self._parse_text("######\n#$@._#\n######\n")

        result = a_star_search(level, initial, lambda _level, _state: 0)

        self.assertEqual(result.status, SearchStatus.FAILURE)
        self.assertGreater(result.expanded_nodes, 0)
        self.assertEqual(result.frontier_size_at_end, 0)
        self.assertIsNone(result.solution_cost)

    def test_cutoff_before_expansion_beyond_configured_limit(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        result = a_star_search(
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

        result = a_star_search(
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

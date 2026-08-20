import unittest
from math import inf

from sia_tp1.domain import apply_move
from sia_tp1.heuristics import (
    minimum_matching_manhattan_distance,
    shortest_push_access_distance,
)
from sia_tp1.model import Direction, Level, State
from sia_tp1.parser import parse_level


class MinimumMatchingManhattanDistanceTest(unittest.TestCase):
    def test_values_match_manual_level_01_calculation(self) -> None:
        level, state = parse_level("levels/level_01.txt")

        values = [minimum_matching_manhattan_distance(level, state)]
        for _ in range(3):
            transition = apply_move(level, state, Direction.RIGHT)
            self.assertIsNotNone(transition)
            state = transition.state
            values.append(minimum_matching_manhattan_distance(level, state))

        self.assertEqual(values, [2, 2, 1, 0])

    def test_chooses_minimum_total_assignment_instead_of_greedy_pairs(
        self,
    ) -> None:
        level = Level(
            walls=frozenset(),
            floors=frozenset({(0, 0), (0, 1), (0, 2), (0, 4), (0, 10)}),
            goals=frozenset({(0, 0), (0, 10)}),
            height=1,
            width=11,
        )
        state = State(
            player=(0, 2),
            boxes=frozenset({(0, 1), (0, 4)}),
        )

        value = minimum_matching_manhattan_distance(level, state)

        self.assertEqual(value, 7)

    def test_rejects_different_box_and_goal_counts(self) -> None:
        level = Level(
            walls=frozenset(),
            floors=frozenset({(0, 0), (0, 1), (0, 2)}),
            goals=frozenset({(0, 2)}),
            height=1,
            width=3,
        )
        state = State(
            player=(0, 0),
            boxes=frozenset({(0, 1), (0, 2)}),
        )

        with self.assertRaisesRegex(ValueError, "number of boxes and goals"):
            minimum_matching_manhattan_distance(level, state)


class ShortestPushAccessDistanceTest(unittest.TestCase):
    def test_values_match_manual_level_01_calculation(self) -> None:
        level, state = parse_level("levels/level_01.txt")

        values = [shortest_push_access_distance(level, state)]
        for _ in range(3):
            transition = apply_move(level, state, Direction.RIGHT)
            self.assertIsNotNone(transition)
            state = transition.state
            values.append(shortest_push_access_distance(level, state))

        self.assertEqual(values, [1, 0, 0, 0])

    def test_uses_shortest_real_path_around_obstacles(self) -> None:
        floors = frozenset(
            {
                (1, 1), (1, 2), (1, 3), (1, 4),
                (2, 1), (2, 4),
                (3, 1), (3, 2), (3, 3), (3, 4),
            }
        )
        level = Level(
            walls=frozenset(),
            floors=floors,
            goals=frozenset({(3, 4)}),
            height=4,
            width=5,
        )
        state = State(player=(3, 2), boxes=frozenset({(1, 3)}))

        value = shortest_push_access_distance(level, state)

        self.assertEqual(value, 4)

    def test_returns_infinity_without_reachable_push_position(self) -> None:
        level = Level(
            walls=frozenset(),
            floors=frozenset({(1, 1), (1, 2), (1, 3), (3, 1), (3, 2)}),
            goals=frozenset({(3, 2)}),
            height=4,
            width=4,
        )
        state = State(player=(3, 1), boxes=frozenset({(1, 2)}))

        value = shortest_push_access_distance(level, state)

        self.assertEqual(value, inf)


if __name__ == "__main__":
    unittest.main()

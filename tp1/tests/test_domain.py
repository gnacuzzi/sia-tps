import unittest

from sia_tp1.domain import apply_move, is_goal
from sia_tp1.model import Direction, Level, State
from sia_tp1.parser import parse_level


class DomainTest(unittest.TestCase):
    def test_free_move_creates_a_new_state_without_push(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        transition = apply_move(level, initial, Direction.RIGHT)

        self.assertIsNotNone(transition)
        self.assertEqual(transition.state.player, (1, 2))
        self.assertEqual(transition.state.boxes, frozenset({(1, 3)}))
        self.assertEqual(transition.direction, Direction.RIGHT)
        self.assertFalse(transition.pushed)
        self.assertEqual(initial.player, (1, 1))
        self.assertEqual(initial.boxes, frozenset({(1, 3)}))

    def test_valid_push_moves_player_and_one_box(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        first = apply_move(level, initial, Direction.RIGHT)

        transition = apply_move(level, first.state, Direction.RIGHT)

        self.assertIsNotNone(transition)
        self.assertEqual(transition.state.player, (1, 3))
        self.assertEqual(transition.state.boxes, frozenset({(1, 4)}))
        self.assertTrue(transition.pushed)
        self.assertEqual(first.state.boxes, frozenset({(1, 3)}))

    def test_wall_returns_no_transition(self) -> None:
        level, initial = parse_level("levels/level_01.txt")

        transition = apply_move(level, initial, Direction.LEFT)

        self.assertIsNone(transition)

    def test_void_returns_no_transition(self) -> None:
        level = Level(
            walls=frozenset(),
            floors=frozenset({(1, 1)}),
            goals=frozenset(),
            height=2,
            width=3,
        )
        state = State(player=(1, 1), boxes=frozenset())

        transition = apply_move(level, state, Direction.RIGHT)

        self.assertIsNone(transition)

    def test_box_against_wall_cannot_be_pushed(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        state = State(player=(1, 4), boxes=frozenset({(1, 5)}))

        transition = apply_move(level, state, Direction.RIGHT)

        self.assertIsNone(transition)

    def test_box_cannot_push_another_box(self) -> None:
        level = Level(
            walls=frozenset(
                {
                    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 0), (1, 4),
                    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                }
            ),
            floors=frozenset({(1, 1), (1, 2), (1, 3)}),
            goals=frozenset({(1, 2), (1, 3)}),
            height=3,
            width=5,
        )
        state = State(
            player=(1, 1), boxes=frozenset({(1, 2), (1, 3)})
        )

        transition = apply_move(level, state, Direction.RIGHT)

        self.assertIsNone(transition)

    def test_box_can_leave_a_goal(self) -> None:
        level = Level(
            walls=frozenset(
                {
                    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 0), (1, 4),
                    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                }
            ),
            floors=frozenset({(1, 1), (1, 2), (1, 3)}),
            goals=frozenset({(1, 2)}),
            height=3,
            width=5,
        )
        state = State(player=(1, 1), boxes=frozenset({(1, 2)}))

        transition = apply_move(level, state, Direction.RIGHT)

        self.assertIsNotNone(transition)
        self.assertEqual(transition.state.boxes, frozenset({(1, 3)}))
        self.assertFalse(is_goal(level, transition.state))

    def test_goal_condition_uses_box_and_goal_positions(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        first = apply_move(level, initial, Direction.RIGHT)
        second = apply_move(level, first.state, Direction.RIGHT)
        third = apply_move(level, second.state, Direction.RIGHT)

        self.assertFalse(is_goal(level, initial))
        self.assertFalse(is_goal(level, first.state))
        self.assertFalse(is_goal(level, second.state))
        self.assertTrue(is_goal(level, third.state))

    def test_three_right_moves_match_the_manual_acceptance_case(self) -> None:
        level, state = parse_level("levels/level_01.txt")
        transitions = []

        for _ in range(3):
            transition = apply_move(level, state, Direction.RIGHT)
            self.assertIsNotNone(transition)
            transitions.append(transition)
            state = transition.state

        self.assertEqual(
            [transition.pushed for transition in transitions],
            [False, True, True],
        )
        self.assertEqual(state.player, (1, 4))
        self.assertEqual(state.boxes, frozenset({(1, 5)}))
        self.assertTrue(is_goal(level, state))
        self.assertIsNone(apply_move(level, state, Direction.RIGHT))


if __name__ == "__main__":
    unittest.main()


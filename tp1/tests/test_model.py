import unittest

from sia_tp1.model import Direction, Level, State


class ModelTest(unittest.TestCase):
    def test_direction_deltas_match_the_agreed_coordinates(self) -> None:
        self.assertEqual(
            [direction.name for direction in Direction],
            ["UP", "DOWN", "LEFT", "RIGHT"],
        )
        self.assertEqual(Direction.UP.delta, (-1, 0))
        self.assertEqual(Direction.DOWN.delta, (1, 0))
        self.assertEqual(Direction.LEFT.delta, (0, -1))
        self.assertEqual(Direction.RIGHT.delta, (0, 1))

    def test_box_order_does_not_change_state_identity(self) -> None:
        first = State(player=(1, 1), boxes=frozenset({(2, 2), (3, 3)}))
        second = State(player=(1, 1), boxes=frozenset({(3, 3), (2, 2)}))

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_level_distinguishes_wall_floor_and_void(self) -> None:
        level = Level(
            walls=frozenset({(0, 0)}),
            floors=frozenset({(0, 1)}),
            goals=frozenset({(0, 1)}),
            height=1,
            width=3,
        )

        self.assertTrue(level.is_wall((0, 0)))
        self.assertTrue(level.is_floor((0, 1)))
        self.assertTrue(level.is_void((0, 2)))


if __name__ == "__main__":
    unittest.main()

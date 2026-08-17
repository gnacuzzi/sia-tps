import tempfile
import unittest
from pathlib import Path

from sia_tp1.domain import apply_move
from sia_tp1.model import Direction
from sia_tp1.parser import parse_level
from sia_tp1.render import render_state


class RenderTest(unittest.TestCase):
    def test_render_initial_level(self) -> None:
        level, state = parse_level("levels/level_01.txt")

        self.assertEqual(
            render_state(level, state),
            "#######\n#@_$_.#\n#######",
        )

    def test_render_player_and_box_on_goals(self) -> None:
        level, state = self._parse_text("######\n#+*$##\n######\n")

        self.assertEqual(render_state(level, state), "######\n#+*$##\n######")

    def test_render_solved_acceptance_level(self) -> None:
        level, state = parse_level("levels/level_01.txt")

        for _ in range(3):
            transition = apply_move(level, state, Direction.RIGHT)
            self.assertIsNotNone(transition)
            state = transition.state

        self.assertEqual(
            render_state(level, state),
            "#######\n#___@*#\n#######",
        )

    def test_render_preserves_irregular_shape_without_trailing_spaces(self) -> None:
        text = "  ###\n###_#\n#@$.#\n#####\n"
        level, state = self._parse_text(text)

        self.assertEqual(render_state(level, state), text.rstrip("\n"))

    def _parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level.txt"
            path.write_text(text, encoding="utf-8")
            return parse_level(path)


if __name__ == "__main__":
    unittest.main()


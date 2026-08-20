import tempfile
import unittest
from pathlib import Path

from sia_tp1.parser import LevelFormatError, parse_level


class ParserTest(unittest.TestCase):
    def test_parse_minimal_level(self) -> None:
        level, state = parse_level("levels/level_01.txt")

        self.assertEqual(state.player, (1, 1))
        self.assertEqual(state.boxes, frozenset({(1, 3)}))
        self.assertEqual(level.goals, frozenset({(1, 5)}))
        self.assertTrue(level.is_floor((1, 1)))
        self.assertTrue(level.is_floor((1, 3)))

    def test_parse_level_02_from_reference_board(self) -> None:
        level, state = parse_level("levels/level_02.txt")

        self.assertEqual(level.height, 10)
        self.assertEqual(level.width, 15)
        self.assertEqual(state.player, (7, 7))
        self.assertEqual(state.boxes, frozenset({(7, 6), (7, 8)}))
        self.assertEqual(level.goals, frozenset({(1, 7), (2, 7)}))
        self.assertTrue(level.is_void((0, 0)))
        self.assertTrue(level.is_floor((3, 7)))

    def test_parse_level_03_from_reference_board(self) -> None:
        level, state = parse_level("levels/level_03.txt")

        self.assertEqual(level.height, 6)
        self.assertEqual(level.width, 8)
        self.assertEqual(state.player, (3, 2))
        self.assertEqual(
            state.boxes,
            frozenset({(2, 4), (3, 3), (3, 5), (4, 2)}),
        )
        self.assertEqual(
            level.goals,
            frozenset({(3, 2), (3, 3), (3, 4), (4, 4)}),
        )
        self.assertTrue(level.is_void((0, 0)))
        self.assertTrue(level.is_floor((1, 2)))

    def test_parse_combined_symbols_by_layer(self) -> None:
        level, state = self._parse_text("######\n#+*$##\n######\n")

        self.assertEqual(state.player, (1, 1))
        self.assertEqual(state.boxes, frozenset({(1, 2), (1, 3)}))
        self.assertEqual(level.goals, frozenset({(1, 1), (1, 2)}))
        self.assertTrue({(1, 1), (1, 2), (1, 3)} <= level.floors)

    def test_parse_irregular_closed_level(self) -> None:
        text = "  ###\n###_#\n#@$.#\n#####\n"
        level, state = self._parse_text(text)

        self.assertTrue(level.is_void((0, 0)))
        self.assertTrue(level.is_wall((0, 2)))
        self.assertTrue(level.is_floor((1, 3)))
        self.assertEqual(state.player, (2, 1))

    def test_reject_level_open_to_exterior(self) -> None:
        text = "####\n#@_ \n#$.#\n####\n"

        with self.assertRaisesRegex(LevelFormatError, "open to the exterior"):
            self._parse_text(text)

    def test_reject_multiple_players(self) -> None:
        text = "#####\n#@@$#\n#_.##\n#####\n"

        with self.assertRaisesRegex(LevelFormatError, "exactly one player"):
            self._parse_text(text)

    def test_reject_different_box_and_goal_counts(self) -> None:
        text = "#####\n#@$$#\n#_.##\n#####\n"

        with self.assertRaisesRegex(LevelFormatError, "same number"):
            self._parse_text(text)

    def test_reject_unknown_symbol(self) -> None:
        text = "#####\n#@?$#\n#_.##\n#####\n"

        with self.assertRaisesRegex(LevelFormatError, "Unknown symbol"):
            self._parse_text(text)

    def _parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level.txt"
            path.write_text(text, encoding="utf-8")
            return parse_level(path)


if __name__ == "__main__":
    unittest.main()

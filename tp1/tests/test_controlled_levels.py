import unittest
from pathlib import Path

from sia_tp1.heuristics import (
    minimum_matching_manhattan_distance,
    shortest_push_access_distance,
)
from sia_tp1.parser import parse_level
from sia_tp1.search import SearchStatus, breadth_first_search


class TestControlledLevelFamilies(unittest.TestCase):
    def test_family_a_only_changes_player_access_distance(self) -> None:
        directory = Path("levels/controlled/family_a")
        expected = {
            "access_near.txt": (0, 5),
            "access_medium.txt": (5, 10),
            "access_far.txt": (11, 16),
        }

        parsed = {
            filename: parse_level(directory / filename)
            for filename in expected
        }
        reference_level, reference_state = parsed["access_near.txt"]

        for filename, (expected_spa, expected_cost) in expected.items():
            with self.subTest(filename=filename):
                level, state = parsed[filename]
                self.assertEqual(level, reference_level)
                self.assertEqual(state.boxes, reference_state.boxes)
                self.assertEqual(
                    shortest_push_access_distance(level, state),
                    expected_spa,
                )
                self.assertEqual(
                    minimum_matching_manhattan_distance(level, state),
                    2,
                )

                result = breadth_first_search(level, state)
                self.assertEqual(result.status, SearchStatus.SUCCESS)
                self.assertEqual(result.solution_cost, expected_cost)
                self.assertEqual(result.solution_pushes, 2)


if __name__ == "__main__":
    unittest.main()

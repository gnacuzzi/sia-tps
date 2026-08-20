import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sia_tp1.parser import parse_level
from sia_tp1.search import Node, SearchResult, SearchStatus, breadth_first_search
from sia_tp1.visualization import save_solution_gif


class SolutionGifTest(unittest.TestCase):
    def test_writes_one_frame_per_solution_state(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        result = breadth_first_search(level, initial)

        with tempfile.TemporaryDirectory() as directory:
            gif_path = Path(directory) / "output" / "videos" / "bfs.gif"

            frame_count = save_solution_gif(
                level,
                result,
                "bfs",
                gif_path,
            )

            self.assertTrue(gif_path.is_file())
            self.assertEqual(frame_count, result.solution_moves + 1)
            with Image.open(gif_path) as gif:
                self.assertEqual(gif.format, "GIF")
                self.assertEqual(gif.n_frames, 4)
                self.assertEqual(gif.info["duration"], 500)
                self.assertEqual(gif.info["loop"], 0)

    def test_rejects_result_without_solution(self) -> None:
        level, _ = parse_level("levels/level_01.txt")
        result = SearchResult(
            status=SearchStatus.FAILURE,
            goal_node=None,
            expanded_nodes=1,
            frontier_size_at_end=0,
            max_frontier_size=1,
            elapsed_seconds=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            gif_path = Path(directory) / "failure.gif"

            with self.assertRaisesRegex(ValueError, "successful search"):
                save_solution_gif(level, result, "bfs", gif_path)

            self.assertFalse(gif_path.exists())

    def test_rejects_success_whose_final_state_is_not_a_goal(self) -> None:
        level, initial = parse_level("levels/level_01.txt")
        root = Node(initial, None, None, 0, 0)
        result = SearchResult(
            status=SearchStatus.SUCCESS,
            goal_node=root,
            expanded_nodes=0,
            frontier_size_at_end=0,
            max_frontier_size=1,
            elapsed_seconds=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            gif_path = Path(directory) / "invalid-success.gif"

            with self.assertRaisesRegex(ValueError, "satisfy the goal"):
                save_solution_gif(level, result, "bfs", gif_path)

            self.assertFalse(gif_path.exists())


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.generate_level_analysis_plots import (
    DEFAULT_SUMMARY,
    generate_plots,
    read_summary,
)


class GenerateLevelAnalysisPlotsTest(unittest.TestCase):
    def test_generates_the_five_narrative_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = generate_plots(
                read_summary(DEFAULT_SUMMARY),
                Path(directory),
                dpi=80,
            )

            self.assertEqual(
                [path.stem for path in generated],
                [
                    "difficulty_dimensions",
                    "path_length_vs_search",
                    "hard_level_sensitivity",
                    "quality_work_tradeoff",
                    "astar_spa_vs_mmm",
                ],
            )
            for path in generated:
                self.assertTrue(path.is_file())
                with Image.open(path) as image:
                    self.assertGreater(image.width, image.height)


if __name__ == "__main__":
    unittest.main()

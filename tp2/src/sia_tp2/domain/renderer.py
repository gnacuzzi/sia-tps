"""Render normalized triangle chromosomes with deterministic alpha composition."""

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw

from .model import Individual, RGB, TriangleGene


def working_size(original_size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_target_images(
    path: Path, max_side: int
) -> Tuple[Image.Image, Image.Image]:
    with Image.open(path) as source:
        original = source.convert("RGB")
    size = working_size(original.size, max_side)
    working = original.resize(size, Image.Resampling.LANCZOS)
    return original, working


def _pixel_vertices(
    triangle: TriangleGene, size: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    width, height = size
    return tuple(
        (round(x * (width - 1)), round(y * (height - 1)))
        for x, y in triangle.vertices
    )


def render_individual(
    individual: Individual,
    *,
    size: Tuple[int, int],
    canvas_rgb: RGB,
) -> Image.Image:
    """Render genes in locus order and return an opaque RGB image."""

    canvas = Image.new("RGBA", size, (*canvas_rgb, 255))
    for triangle in individual.chromosome:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(
            _pixel_vertices(triangle, size), fill=triangle.color
        )
        canvas = Image.alpha_composite(canvas, layer)
    return canvas.convert("RGB")

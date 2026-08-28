"""Reproducible random initialization for triangle chromosomes."""

import random
from typing import Tuple

from .model import Individual, TriangleGene


def create_random_triangle(
    rng: random.Random,
    *,
    alpha_range: Tuple[int, int],
) -> TriangleGene:
    """Sample one valid triangle, retrying the negligible degenerate case."""

    while True:
        vertices = tuple((rng.random(), rng.random()) for _ in range(3))
        color = (
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(alpha_range[0], alpha_range[1]),
        )
        try:
            return TriangleGene(vertices=vertices, color=color)
        except ValueError as error:
            if "positive area" not in str(error):
                raise


def create_random_individual(
    rng: random.Random,
    *,
    triangle_count: int,
    alpha_range: Tuple[int, int],
) -> Individual:
    chromosome = tuple(
        create_random_triangle(rng, alpha_range=alpha_range)
        for _ in range(triangle_count)
    )
    return Individual(chromosome=chromosome)


def create_random_population(
    *,
    population_size: int,
    triangle_count: int,
    alpha_range: Tuple[int, int],
    seed: int,
) -> Tuple[Individual, ...]:
    rng = random.Random(seed)
    return tuple(
        create_random_individual(
            rng,
            triangle_count=triangle_count,
            alpha_range=alpha_range,
        )
        for _ in range(population_size)
    )

"""Genotypic diversity for populations of triangle chromosomes."""

from typing import Sequence, Tuple

from .model import Individual


def triangle_population_diversity(population: Sequence[Individual]) -> float:
    """Average normalized allele distance over every unordered pair."""

    if len(population) < 2:
        return 0.0
    vectors = tuple(_normalized_alleles(individual) for individual in population)
    dimension_count = len(vectors[0])
    if any(len(vector) != dimension_count for vector in vectors):
        raise ValueError("all individuals must have equal chromosome lengths")

    pair_count = len(vectors) * (len(vectors) - 1) // 2
    difference_sum = 0.0
    for dimension in range(dimension_count):
        values = sorted(vector[dimension] for vector in vectors)
        difference_sum += sum(
            (2 * index - len(values) + 1) * value
            for index, value in enumerate(values)
        )
    return difference_sum / (pair_count * dimension_count)


def _normalized_alleles(individual: Individual) -> Tuple[float, ...]:
    values = []
    for triangle in individual.chromosome:
        values.extend(coordinate for vertex in triangle.vertices for coordinate in vertex)
        values.extend(channel / 255.0 for channel in triangle.color)
    return tuple(values)

"""Crossover and mutation operators for triangle chromosomes."""

import random
from typing import Tuple

from .model import Individual, TriangleGene


def uniform_crossover(
    first: Individual,
    second: Individual,
    *,
    probability: float,
    swap_probability: float,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """Exchange whole genes at matching loci between two parents."""

    if len(first.chromosome) != len(second.chromosome):
        raise ValueError("parents must have equal chromosome lengths")
    if rng.random() >= probability:
        return (
            Individual(chromosome=first.chromosome),
            Individual(chromosome=second.chromosome),
        )

    first_child = []
    second_child = []
    for first_gene, second_gene in zip(first.chromosome, second.chromosome):
        if rng.random() < swap_probability:
            first_child.append(second_gene)
            second_child.append(first_gene)
        else:
            first_child.append(first_gene)
            second_child.append(second_gene)
    return (
        Individual(chromosome=tuple(first_child)),
        Individual(chromosome=tuple(second_child)),
    )


def multigene_uniform_local_mutation(
    individual: Individual,
    *,
    probability: float,
    position_delta: float,
    color_delta: int,
    alpha_delta: int,
    alpha_range: Tuple[int, int],
    rng: random.Random,
) -> Individual:
    """Independently mutate each gene by changing one of seven properties."""

    chromosome = tuple(
        _mutate_triangle_local(
            gene,
            position_delta=position_delta,
            color_delta=color_delta,
            alpha_delta=alpha_delta,
            alpha_range=alpha_range,
            rng=rng,
        )
        if rng.random() < probability
        else gene
        for gene in individual.chromosome
    )
    return Individual(chromosome=chromosome)


def _mutate_triangle_local(
    triangle: TriangleGene,
    *,
    position_delta: float,
    color_delta: int,
    alpha_delta: int,
    alpha_range: Tuple[int, int],
    rng: random.Random,
) -> TriangleGene:
    property_index = rng.randrange(7)
    if property_index < 3:
        return _mutate_vertex(
            triangle, property_index, position_delta=position_delta, rng=rng
        )

    channel_index = property_index - 3
    color = list(triangle.color)
    if channel_index < 3:
        color[channel_index] = _different_local_integer(
            color[channel_index], 0, 255, color_delta, rng
        )
    else:
        color[3] = _different_local_integer(
            color[3], alpha_range[0], alpha_range[1], alpha_delta, rng
        )
    return TriangleGene(vertices=triangle.vertices, color=tuple(color))


def _mutate_vertex(
    triangle: TriangleGene,
    vertex_index: int,
    *,
    position_delta: float,
    rng: random.Random,
) -> TriangleGene:
    while True:
        x, y = triangle.vertices[vertex_index]
        changed = (
            min(1.0, max(0.0, x + rng.uniform(-position_delta, position_delta))),
            min(1.0, max(0.0, y + rng.uniform(-position_delta, position_delta))),
        )
        if changed == (x, y):
            continue
        vertices = list(triangle.vertices)
        vertices[vertex_index] = changed
        try:
            return TriangleGene(vertices=tuple(vertices), color=triangle.color)
        except ValueError as error:
            if "positive area" not in str(error):
                raise


def _different_local_integer(
    current: int,
    domain_minimum: int,
    domain_maximum: int,
    delta: int,
    rng: random.Random,
) -> int:
    minimum = max(domain_minimum, current - delta)
    maximum = min(domain_maximum, current + delta)
    alternatives = maximum - minimum
    if alternatives < 1:
        raise ValueError("the configured domain has no alternative mutation value")
    sampled = rng.randrange(alternatives)
    value = minimum + sampled
    return value + 1 if value >= current else value

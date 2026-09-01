"""Crossover and mutation operators for triangle chromosomes."""

import random
from typing import Optional, Tuple

from .model import Individual, TriangleGene


def crossover_individuals(
    first: Individual,
    second: Individual,
    *,
    method: str,
    probability: float,
    swap_probability: Optional[float],
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    if method == "one_point":
        return one_point_crossover(
            first, second, probability=probability, rng=rng
        )
    if method == "uniform":
        if swap_probability is None:
            raise ValueError("uniform crossover requires swap_probability")
        return uniform_crossover(
            first,
            second,
            probability=probability,
            swap_probability=swap_probability,
            rng=rng,
        )
    raise ValueError(f"unsupported crossover method: {method}")


def one_point_crossover(
    first: Individual,
    second: Individual,
    *,
    probability: float,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """Exchange chromosome suffixes after one internal locus boundary."""

    _validate_parents(first, second)
    if len(first.chromosome) < 2:
        raise ValueError("one-point crossover requires at least two genes")
    if rng.random() >= probability:
        return _unevaluated_copies(first, second)
    point = rng.randrange(1, len(first.chromosome))
    return (
        Individual(
            chromosome=first.chromosome[:point] + second.chromosome[point:]
        ),
        Individual(
            chromosome=second.chromosome[:point] + first.chromosome[point:]
        ),
    )


def uniform_crossover(
    first: Individual,
    second: Individual,
    *,
    probability: float,
    swap_probability: float,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """Exchange whole genes at matching loci between two parents."""

    _validate_parents(first, second)
    if rng.random() >= probability:
        return _unevaluated_copies(first, second)

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


def mutate_individual(
    individual: Individual,
    *,
    method: str,
    probability: float,
    allele_mode: str,
    position_delta: Optional[float],
    color_delta: Optional[int],
    alpha_delta: Optional[int],
    alpha_range: Tuple[int, int],
    rng: random.Random,
) -> Individual:
    """Dispatch one mutation method and one allele-change mode."""

    def mutate_gene(gene: TriangleGene) -> TriangleGene:
        return _mutate_triangle(
            gene,
            mode=allele_mode,
            position_delta=position_delta,
            color_delta=color_delta,
            alpha_delta=alpha_delta,
            alpha_range=alpha_range,
            rng=rng,
        )

    if method == "single_gene":
        chromosome = list(individual.chromosome)
        if rng.random() < probability:
            index = rng.randrange(len(chromosome))
            chromosome[index] = mutate_gene(chromosome[index])
        return Individual(chromosome=tuple(chromosome))
    if method == "multigene_uniform":
        return Individual(
            chromosome=tuple(
                mutate_gene(gene) if rng.random() < probability else gene
                for gene in individual.chromosome
            )
        )
    raise ValueError(f"unsupported mutation method: {method}")


def _mutate_triangle(
    triangle: TriangleGene,
    *,
    mode: str,
    position_delta: Optional[float],
    color_delta: Optional[int],
    alpha_delta: Optional[int],
    alpha_range: Tuple[int, int],
    rng: random.Random,
) -> TriangleGene:
    property_index = rng.randrange(7)
    if property_index < 3:
        return _mutate_vertex(
            triangle,
            property_index,
            mode=mode,
            position_delta=position_delta,
            rng=rng,
        )

    channel_index = property_index - 3
    color = list(triangle.color)
    if channel_index < 3:
        color[channel_index] = _mutate_integer(
            color[channel_index],
            domain_minimum=0,
            domain_maximum=255,
            mode=mode,
            delta=color_delta,
            rng=rng,
        )
    else:
        color[3] = _mutate_integer(
            color[3],
            domain_minimum=alpha_range[0],
            domain_maximum=alpha_range[1],
            mode=mode,
            delta=alpha_delta,
            rng=rng,
        )
    return TriangleGene(vertices=triangle.vertices, color=tuple(color))


def _mutate_vertex(
    triangle: TriangleGene,
    vertex_index: int,
    *,
    mode: str,
    position_delta: Optional[float],
    rng: random.Random,
) -> TriangleGene:
    while True:
        x, y = triangle.vertices[vertex_index]
        if mode == "local_delta":
            if position_delta is None:
                raise ValueError("local mutation requires position_delta")
            changed = (
                min(1.0, max(0.0, x + rng.uniform(-position_delta, position_delta))),
                min(1.0, max(0.0, y + rng.uniform(-position_delta, position_delta))),
            )
        elif mode == "global_resample":
            changed = (rng.random(), rng.random())
        else:
            raise ValueError(f"unsupported allele-change mode: {mode}")
        if changed == (x, y):
            continue
        vertices = list(triangle.vertices)
        vertices[vertex_index] = changed
        try:
            return TriangleGene(vertices=tuple(vertices), color=triangle.color)
        except ValueError as error:
            if "positive area" not in str(error):
                raise


def _mutate_integer(
    current: int,
    *,
    domain_minimum: int,
    domain_maximum: int,
    mode: str,
    delta: Optional[int],
    rng: random.Random,
) -> int:
    if mode == "local_delta":
        if delta is None:
            raise ValueError("local mutation requires a channel delta")
        minimum = max(domain_minimum, current - delta)
        maximum = min(domain_maximum, current + delta)
    elif mode == "global_resample":
        minimum = domain_minimum
        maximum = domain_maximum
    else:
        raise ValueError(f"unsupported allele-change mode: {mode}")
    alternatives = maximum - minimum
    if alternatives < 1:
        raise ValueError("the configured domain has no alternative mutation value")
    sampled = rng.randrange(alternatives)
    value = minimum + sampled
    return value + 1 if value >= current else value


def _validate_parents(first: Individual, second: Individual) -> None:
    if len(first.chromosome) != len(second.chromosome):
        raise ValueError("parents must have equal chromosome lengths")


def _unevaluated_copies(
    first: Individual, second: Individual
) -> Tuple[Individual, Individual]:
    return (
        Individual(chromosome=first.chromosome),
        Individual(chromosome=second.chromosome),
    )

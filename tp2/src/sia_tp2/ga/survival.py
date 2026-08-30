"""Survivor-selection operators independent from the problem domain."""

from typing import Callable, Sequence, Tuple, TypeVar


T = TypeVar("T")


def additive_elite_survival(
    population: Sequence[T],
    offspring: Sequence[T],
    *,
    population_size: int,
    fitness: Callable[[T], float],
) -> Tuple[T, ...]:
    """Keep the fittest individuals from parents and offspring together."""

    candidates = tuple(population) + tuple(offspring)
    if population_size < 1 or population_size > len(candidates):
        raise ValueError("population_size must fit within the candidate pool")
    return tuple(
        sorted(candidates, key=fitness, reverse=True)[:population_size]
    )

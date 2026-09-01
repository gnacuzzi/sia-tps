"""Survival strategies independent from the problem domain."""

import random
from typing import Callable, Sequence, Tuple, TypeVar


T = TypeVar("T")
Selector = Callable[[Sequence[T], int, int, random.Random], Sequence[T]]


def select_survivors(
    population: Sequence[T],
    offspring: Sequence[T],
    *,
    population_size: int,
    strategy: str,
    generation: int,
    selector: Selector[T],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Apply additive or exclusive replacement with a configured selector."""

    if population_size < 1:
        raise ValueError("population_size must be positive")
    if strategy == "additive":
        survivors = tuple(
            selector((*population, *offspring), population_size, generation, rng)
        )
    elif strategy == "exclusive":
        if len(offspring) > population_size:
            survivors = tuple(
                selector(offspring, population_size, generation, rng)
            )
        else:
            retained_parents = tuple(
                selector(
                    population,
                    population_size - len(offspring),
                    generation,
                    rng,
                )
            )
            survivors = tuple(offspring) + retained_parents
    else:
        raise ValueError(f"unsupported survival strategy: {strategy}")

    if len(survivors) != population_size:
        raise ValueError("survival selector returned an unexpected count")
    return survivors

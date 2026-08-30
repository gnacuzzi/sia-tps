"""Parent-selection operators independent from the problem domain."""

import random
from typing import Callable, Sequence, Tuple, TypeVar


T = TypeVar("T")


def deterministic_tournament(
    population: Sequence[T],
    *,
    count: int,
    tournament_size: int,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Select ``count`` parents by independent tournaments without replacement."""

    if count < 0:
        raise ValueError("count cannot be negative")
    if not 2 <= tournament_size <= len(population):
        raise ValueError("tournament_size must be between 2 and population size")

    selected = []
    for _ in range(count):
        participants = rng.sample(population, tournament_size)
        selected.append(max(participants, key=fitness))
    return tuple(selected)

"""Selection operators independent from the problem domain."""

import bisect
import math
import random
from typing import Callable, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T")


def select_population(
    population: Sequence[T],
    *,
    count: int,
    method: str,
    params: Mapping[str, object],
    generation: int,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Dispatch one configured selector through a common interface."""

    if method == "elite":
        return elite_selection(population, count=count, fitness=fitness)
    if method == "roulette":
        return roulette_selection(population, count=count, fitness=fitness, rng=rng)
    if method == "universal":
        return universal_selection(population, count=count, fitness=fitness, rng=rng)
    if method == "ranking":
        return ranking_selection(population, count=count, fitness=fitness, rng=rng)
    if method == "boltzmann":
        return boltzmann_selection(
            population,
            count=count,
            generation=generation,
            initial_temperature=float(params["initial_temperature"]),
            final_temperature=float(params["final_temperature"]),
            decay_rate=float(params["decay_rate"]),
            fitness=fitness,
            rng=rng,
        )
    if method == "tournament_deterministic":
        return deterministic_tournament(
            population,
            count=count,
            tournament_size=int(params["tournament_size"]),
            fitness=fitness,
            rng=rng,
        )
    if method == "tournament_probabilistic":
        return probabilistic_tournament(
            population,
            count=count,
            threshold=float(params["threshold"]),
            fitness=fitness,
            rng=rng,
        )
    raise ValueError(f"unsupported selection method: {method}")


def elite_selection(
    population: Sequence[T],
    *,
    count: int,
    fitness: Callable[[T], float],
) -> Tuple[T, ...]:
    """Select by descending fitness, cycling if count exceeds pool size."""

    _validate_request(population, count)
    ranked = sorted(population, key=fitness, reverse=True)
    return tuple(ranked[index % len(ranked)] for index in range(count))


def roulette_selection(
    population: Sequence[T],
    *,
    count: int,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Sample with replacement proportionally to raw fitness."""

    return _weighted_sample(
        population,
        weights=[fitness(item) for item in population],
        count=count,
        rng=rng,
        universal=False,
    )


def universal_selection(
    population: Sequence[T],
    *,
    count: int,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Use one random start and equally spaced pointers over fitness mass."""

    return _weighted_sample(
        population,
        weights=[fitness(item) for item in population],
        count=count,
        rng=rng,
        universal=True,
    )


def ranking_selection(
    population: Sequence[T],
    *,
    count: int,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Apply roulette to rank-based pseudo-fitness instead of raw fitness."""

    _validate_request(population, count)
    ranked = sorted(population, key=fitness, reverse=True)
    weights = [len(ranked) - rank for rank in range(1, len(ranked) + 1)]
    return _weighted_sample(
        ranked, weights=weights, count=count, rng=rng, universal=False
    )


def boltzmann_temperature(
    generation: int,
    *,
    initial_temperature: float,
    final_temperature: float,
    decay_rate: float,
) -> float:
    return final_temperature + (initial_temperature - final_temperature) * math.exp(
        -decay_rate * generation
    )


def boltzmann_selection(
    population: Sequence[T],
    *,
    count: int,
    generation: int,
    initial_temperature: float,
    final_temperature: float,
    decay_rate: float,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Apply roulette to stable Boltzmann-scaled fitness values."""

    _validate_request(population, count)
    temperature = boltzmann_temperature(
        generation,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature,
        decay_rate=decay_rate,
    )
    values = [fitness(item) for item in population]
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    return _weighted_sample(
        population, weights=weights, count=count, rng=rng, universal=False
    )


def deterministic_tournament(
    population: Sequence[T],
    *,
    count: int,
    tournament_size: int,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Select each winner from a random tournament of configurable size."""

    _validate_request(population, count)
    if not 2 <= tournament_size <= len(population):
        raise ValueError("tournament_size must be between 2 and population size")
    selected = []
    for _ in range(count):
        participants = rng.sample(population, tournament_size)
        selected.append(max(participants, key=fitness))
    return tuple(selected)


def probabilistic_tournament(
    population: Sequence[T],
    *,
    count: int,
    threshold: float,
    fitness: Callable[[T], float],
    rng: random.Random,
) -> Tuple[T, ...]:
    """Choose the fitter of two with probability threshold, otherwise the worse."""

    _validate_request(population, count)
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.5, 1]")
    selected = []
    for _ in range(count):
        first, second = rng.sample(population, 2)
        better, worse = sorted((first, second), key=fitness, reverse=True)
        selected.append(better if rng.random() < threshold else worse)
    return tuple(selected)


def _weighted_sample(
    population: Sequence[T],
    *,
    weights: Sequence[float],
    count: int,
    rng: random.Random,
    universal: bool,
) -> Tuple[T, ...]:
    _validate_request(population, count)
    if len(weights) != len(population):
        raise ValueError("weights and population must have equal lengths")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("selection weights must be finite and non-negative")
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("at least one selection weight must be positive")
    if count == 0:
        return ()

    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running)
    cumulative[-1] = total

    if universal:
        step = total / count
        start = rng.random() * step
        pointers = (start + index * step for index in range(count))
    else:
        pointers = (rng.random() * total for _ in range(count))
    return tuple(
        population[min(bisect.bisect_right(cumulative, pointer), len(population) - 1)]
        for pointer in pointers
    )


def _validate_request(population: Sequence[T], count: int) -> None:
    if not population:
        raise ValueError("selection population cannot be empty")
    if count < 0:
        raise ValueError("count cannot be negative")

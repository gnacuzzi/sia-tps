"""Domain-independent vertical genetic-algorithm loop."""

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class EvolutionLimits:
    max_generations: int
    target_error: Optional[float] = None
    stagnation_patience: Optional[int] = None
    min_improvement: float = 0.0
    max_seconds: Optional[float] = None


@dataclass(frozen=True)
class GenerationMetrics:
    generation: int
    evaluations: int
    best_error: float
    mean_error: float
    median_error: float
    error_stddev: float
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    fitness_stddev: float
    diversity: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "evaluations": self.evaluations,
            "best_error": self.best_error,
            "mean_error": self.mean_error,
            "median_error": self.median_error,
            "error_stddev": self.error_stddev,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "median_fitness": self.median_fitness,
            "fitness_stddev": self.fitness_stddev,
            "diversity": self.diversity,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class EvolutionResult(Generic[T]):
    final_population: Tuple[T, ...]
    best: T
    best_generation: int
    final_generation: int
    evaluations: int
    stop_reason: str
    metrics: Tuple[GenerationMetrics, ...]


def evolve(
    initial_population: Sequence[T],
    *,
    offspring_count: int,
    seed: int,
    limits: EvolutionLimits,
    evaluate: Callable[[T], T],
    error: Callable[[T], float],
    fitness: Callable[[T], float],
    select_parents: Callable[
        [Sequence[T], int, int, random.Random], Sequence[T]
    ],
    crossover: Callable[[T, T, random.Random], Tuple[T, T]],
    mutate: Callable[[T, random.Random], T],
    survive: Callable[
        [Sequence[T], Sequence[T], int, random.Random], Sequence[T]
    ],
    diversity: Callable[[Sequence[T]], float],
    on_generation: Optional[
        Callable[[GenerationMetrics, Sequence[T], T], None]
    ] = None,
    clock: Callable[[], float] = time.monotonic,
) -> EvolutionResult[T]:
    """Run evolution while delegating every domain-specific operation."""

    _validate_arguments(initial_population, offspring_count, limits)
    rng = random.Random(seed)
    started_at = clock()
    population = tuple(evaluate(item) for item in initial_population)
    evaluations = len(population)
    best = min(population, key=error)
    best_generation = 0
    last_significant_error = error(best)
    stagnant_generations = 0
    records = [
        _measure(
            population,
            generation=0,
            evaluations=evaluations,
            elapsed_seconds=clock() - started_at,
            error=error,
            fitness=fitness,
            diversity=diversity,
        )
    ]
    if on_generation is not None:
        on_generation(records[-1], population, best)

    immediate_reason = _initial_stop_reason(
        best_error=error(best), limits=limits, elapsed=clock() - started_at
    )
    if immediate_reason is not None:
        return _result(
            population, best, best_generation, records, immediate_reason
        )

    stop_reason = "max_generations"
    for generation in range(1, limits.max_generations + 1):
        parents = tuple(
            select_parents(population, offspring_count, generation, rng)
        )
        if len(parents) != offspring_count:
            raise ValueError("parent selection returned an unexpected count")

        children = []
        for index in range(0, offspring_count, 2):
            first, second = crossover(parents[index], parents[index + 1], rng)
            children.extend((mutate(first, rng), mutate(second, rng)))

        evaluated_children = tuple(evaluate(child) for child in children)
        evaluations += len(evaluated_children)
        population = tuple(
            survive(population, evaluated_children, generation, rng)
        )
        if len(population) != len(initial_population):
            raise ValueError("survival changed the configured population size")

        generation_best = min(population, key=error)
        if error(generation_best) < error(best):
            best = generation_best
            best_generation = generation

        improvement = last_significant_error - error(best)
        if improvement > 0.0 and improvement >= limits.min_improvement:
            last_significant_error = error(best)
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        elapsed = clock() - started_at
        records.append(
            _measure(
                population,
                generation=generation,
                evaluations=evaluations,
                elapsed_seconds=elapsed,
                error=error,
                fitness=fitness,
                diversity=diversity,
            )
        )
        if on_generation is not None:
            on_generation(records[-1], population, best)

        reason = _stop_reason(
            best_error=error(best),
            generation=generation,
            stagnant_generations=stagnant_generations,
            elapsed=elapsed,
            limits=limits,
        )
        if reason is not None:
            stop_reason = reason
            break

    return _result(population, best, best_generation, records, stop_reason)


def _validate_arguments(
    population: Sequence[T], offspring_count: int, limits: EvolutionLimits
) -> None:
    if len(population) < 2:
        raise ValueError("initial population must contain at least two individuals")
    if offspring_count < 1 or offspring_count % 2 != 0:
        raise ValueError("offspring_count must be positive and even")
    if limits.max_generations < 1:
        raise ValueError("max_generations must be positive")
    if limits.stagnation_patience is not None and limits.stagnation_patience < 1:
        raise ValueError("stagnation_patience must be positive")
    if limits.min_improvement < 0.0:
        raise ValueError("min_improvement cannot be negative")


def _measure(
    population: Sequence[T],
    *,
    generation: int,
    evaluations: int,
    elapsed_seconds: float,
    error: Callable[[T], float],
    fitness: Callable[[T], float],
    diversity: Callable[[Sequence[T]], float],
) -> GenerationMetrics:
    errors = [error(item) for item in population]
    fitnesses = [fitness(item) for item in population]
    diversity_value = diversity(population)
    if not math.isfinite(diversity_value) or not 0.0 <= diversity_value <= 1.0:
        raise ValueError("diversity must be finite and in [0, 1]")
    return GenerationMetrics(
        generation=generation,
        evaluations=evaluations,
        best_error=min(errors),
        mean_error=statistics.fmean(errors),
        median_error=statistics.median(errors),
        error_stddev=statistics.pstdev(errors),
        best_fitness=max(fitnesses),
        mean_fitness=statistics.fmean(fitnesses),
        median_fitness=statistics.median(fitnesses),
        fitness_stddev=statistics.pstdev(fitnesses),
        diversity=diversity_value,
        elapsed_seconds=elapsed_seconds,
    )


def _initial_stop_reason(
    *, best_error: float, limits: EvolutionLimits, elapsed: float
) -> Optional[str]:
    if limits.target_error is not None and best_error <= limits.target_error:
        return "target_error"
    if limits.max_seconds is not None and elapsed >= limits.max_seconds:
        return "max_seconds"
    return None


def _stop_reason(
    *,
    best_error: float,
    generation: int,
    stagnant_generations: int,
    elapsed: float,
    limits: EvolutionLimits,
) -> Optional[str]:
    if limits.target_error is not None and best_error <= limits.target_error:
        return "target_error"
    if limits.max_seconds is not None and elapsed >= limits.max_seconds:
        return "max_seconds"
    if (
        limits.stagnation_patience is not None
        and stagnant_generations >= limits.stagnation_patience
    ):
        return "stagnation"
    if generation >= limits.max_generations:
        return "max_generations"
    return None


def _result(
    population: Tuple[T, ...],
    best: T,
    best_generation: int,
    records: Sequence[GenerationMetrics],
    stop_reason: str,
) -> EvolutionResult[T]:
    return EvolutionResult(
        final_population=population,
        best=best,
        best_generation=best_generation,
        final_generation=records[-1].generation,
        evaluations=records[-1].evaluations,
        stop_reason=stop_reason,
        metrics=tuple(records),
    )

from dataclasses import dataclass, replace

from sia_tp2.ga.engine import EvolutionLimits, evolve


@dataclass(frozen=True)
class Candidate:
    value: int
    error: float = 0.0
    fitness: float = 0.0


def test_engine_runs_stages_in_order_and_improves_controlled_problem() -> None:
    events = []

    def evaluate(candidate):
        events.append("evaluate")
        return replace(
            candidate,
            error=(10 - candidate.value) / 10,
            fitness=candidate.value / 10,
        )

    def select(population, count, rng):
        del rng
        events.append("select")
        return (max(population, key=lambda item: item.fitness),) * count

    def crossover(first, second, rng):
        del rng
        events.append("crossover")
        return first, second

    def mutate(candidate, rng):
        del rng
        events.append("mutate")
        return Candidate(candidate.value + 1)

    def survive(population, children, rng):
        del rng
        events.append("survive")
        return tuple(
            sorted(
                (*population, *children),
                key=lambda item: item.fitness,
                reverse=True,
            )[: len(population)]
        )

    result = evolve(
        (Candidate(1), Candidate(2)),
        offspring_count=2,
        seed=7,
        limits=EvolutionLimits(max_generations=2),
        evaluate=evaluate,
        error=lambda item: item.error,
        fitness=lambda item: item.fitness,
        select_parents=select,
        crossover=crossover,
        mutate=mutate,
        survive=survive,
        diversity=lambda population: 0.0,
    )

    assert result.best.value == 4
    assert result.best_generation == 2
    assert result.final_generation == 2
    assert result.evaluations == 6
    assert result.stop_reason == "max_generations"
    assert [row.best_error for row in result.metrics] == [0.8, 0.7, 0.6]
    assert events == [
        "evaluate",
        "evaluate",
        "select",
        "crossover",
        "mutate",
        "mutate",
        "evaluate",
        "evaluate",
        "survive",
        "select",
        "crossover",
        "mutate",
        "mutate",
        "evaluate",
        "evaluate",
        "survive",
    ]


def test_engine_stops_after_configured_stagnation_window() -> None:
    def evaluate(candidate):
        return replace(candidate, error=0.5, fitness=0.5)

    result = evolve(
        (Candidate(1), Candidate(2)),
        offspring_count=2,
        seed=0,
        limits=EvolutionLimits(
            max_generations=10,
            stagnation_patience=2,
            min_improvement=0.01,
        ),
        evaluate=evaluate,
        error=lambda item: item.error,
        fitness=lambda item: item.fitness,
        select_parents=lambda population, count, rng: population,
        crossover=lambda first, second, rng: (first, second),
        mutate=lambda item, rng: Candidate(item.value),
        survive=lambda population, children, rng: children,
        diversity=lambda population: 0.0,
    )

    assert result.stop_reason == "stagnation"
    assert result.final_generation == 2
    assert result.evaluations == 6


def test_engine_keeps_global_best_even_if_survival_discards_it() -> None:
    def evaluate(candidate):
        return replace(
            candidate,
            error=candidate.value / 10,
            fitness=(10 - candidate.value) / 10,
        )

    result = evolve(
        (Candidate(1), Candidate(2)),
        offspring_count=2,
        seed=0,
        limits=EvolutionLimits(max_generations=1),
        evaluate=evaluate,
        error=lambda item: item.error,
        fitness=lambda item: item.fitness,
        select_parents=lambda population, count, rng: population,
        crossover=lambda first, second, rng: (first, second),
        mutate=lambda item, rng: Candidate(9),
        survive=lambda population, children, rng: children,
        diversity=lambda population: 0.0,
    )

    assert result.final_population == (
        Candidate(9, error=0.9, fitness=0.1),
        Candidate(9, error=0.9, fitness=0.1),
    )
    assert result.best == Candidate(1, error=0.1, fitness=0.9)
    assert result.best_generation == 0

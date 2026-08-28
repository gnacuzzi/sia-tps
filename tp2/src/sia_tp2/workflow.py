"""Phase 1 workflow: create and evaluate one reproducible random population."""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sia_tp2.config import AppConfig
from sia_tp2.domain.fitness import evaluate_individual
from sia_tp2.domain.initialization import create_random_population
from sia_tp2.domain.model import Individual
from sia_tp2.domain.renderer import load_target_images, render_individual
from sia_tp2.reporting.serialization import (
    create_run_directory,
    write_phase1_artifacts,
)


@dataclass(frozen=True)
class RandomPopulationResult:
    run_directory: Path
    best: Individual
    best_index: int
    working_size: Tuple[int, int]
    original_size: Tuple[int, int]


def render_random_population(config: AppConfig) -> RandomPopulationResult:
    original_target, working_target = load_target_images(
        config.input.image, config.input.working_max_side
    )
    population = create_random_population(
        population_size=config.genetic.population_size,
        triangle_count=config.representation.triangle_count,
        alpha_range=config.representation.alpha_range,
        seed=config.run.seed,
    )

    evaluated: List[Individual] = []
    working_images = []
    for individual in population:
        result, generated = evaluate_individual(
            individual,
            target=working_target,
            canvas_rgb=config.representation.canvas_rgb,
            epsilon=config.fitness.epsilon,
        )
        evaluated.append(result)
        working_images.append(generated)

    best_index = min(
        range(len(evaluated)),
        key=lambda index: _required_error(evaluated[index]),
    )
    best = evaluated[best_index]
    if config.output.render_original_size:
        best_image = render_individual(
            best,
            size=original_target.size,
            canvas_rgb=config.representation.canvas_rgb,
        )
    else:
        best_image = working_images[best_index]

    errors = [_required_error(individual) for individual in evaluated]
    fitnesses = [_required_fitness(individual) for individual in evaluated]
    metrics = {
        "generation": 0,
        "evaluations": len(evaluated),
        "best_error": min(errors),
        "mean_error": statistics.fmean(errors),
        "median_error": statistics.median(errors),
        "error_stddev": statistics.pstdev(errors),
        "best_fitness": max(fitnesses),
        "mean_fitness": statistics.fmean(fitnesses),
        "median_fitness": statistics.median(fitnesses),
        "fitness_stddev": statistics.pstdev(fitnesses),
    }
    metadata = {
        "mode": "render_random",
        "seed": config.run.seed,
        "stop_reason": "initial_population_only",
        "final_generation": 0,
        "best_generation": 0,
        "best_index": best_index,
        "best_error": _required_error(best),
        "best_fitness": _required_fitness(best),
        "population_size": config.genetic.population_size,
        "triangle_count": config.representation.triangle_count,
        "working_size": list(working_target.size),
        "original_size": list(original_target.size),
        "output_size": list(best_image.size),
    }

    run_directory = create_run_directory(
        config.output.directory, f"random-seed-{config.run.seed}"
    )
    write_phase1_artifacts(
        run_directory=run_directory,
        effective_config=config.effective_dict(),
        metadata=metadata,
        metrics=metrics,
        best=best,
        best_image=best_image,
    )
    return RandomPopulationResult(
        run_directory=run_directory,
        best=best,
        best_index=best_index,
        working_size=working_target.size,
        original_size=original_target.size,
    )


def _required_error(individual: Individual) -> float:
    if individual.error is None:
        raise ValueError("individual has not been evaluated")
    return individual.error


def _required_fitness(individual: Individual) -> float:
    if individual.fitness is None:
        raise ValueError("individual has not been evaluated")
    return individual.fitness


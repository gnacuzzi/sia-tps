"""Image workflows for random baselines and genetic evolution."""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sia_tp2.config import AppConfig
from sia_tp2.domain.fitness import evaluate_individual
from sia_tp2.domain.diversity import triangle_population_diversity
from sia_tp2.domain.initialization import create_random_population
from sia_tp2.domain.model import Individual
from sia_tp2.domain.operators import (
    crossover_individuals,
    mutate_individual,
)
from sia_tp2.domain.renderer import load_target_images, render_individual
from sia_tp2.ga.engine import EvolutionLimits, EvolutionResult, evolve
from sia_tp2.ga.selection import select_population
from sia_tp2.ga.survival import select_survivors
from sia_tp2.reporting.serialization import (
    create_run_directory,
    write_checkpoint,
    write_run_artifacts,
)


@dataclass(frozen=True)
class RandomPopulationResult:
    run_directory: Path
    best: Individual
    best_index: int
    working_size: Tuple[int, int]
    original_size: Tuple[int, int]


@dataclass(frozen=True)
class ImageEvolutionResult:
    run_directory: Path
    evolution: EvolutionResult[Individual]
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
    write_run_artifacts(
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


def evolve_image(config: AppConfig) -> ImageEvolutionResult:
    """Evolve triangle images with operators selected entirely from config."""

    original_target, working_target = load_target_images(
        config.input.image, config.input.working_max_side
    )
    initial_population = create_random_population(
        population_size=config.genetic.population_size,
        triangle_count=config.representation.triangle_count,
        alpha_range=config.representation.alpha_range,
        seed=config.run.seed,
    )
    run_directory = create_run_directory(
        config.output.directory, f"evolution-seed-{config.run.seed}"
    )

    def evaluator(individual: Individual) -> Individual:
        evaluated, _ = evaluate_individual(
            individual,
            target=working_target,
            canvas_rgb=config.representation.canvas_rgb,
            epsilon=config.fitness.epsilon,
        )
        return evaluated

    def parent_selector(population, count, generation, rng):
        selection = config.genetic.parent_selection
        return select_population(
            population,
            count=count,
            method=selection.method,
            params=selection.params,
            generation=generation,
            fitness=_required_fitness,
            rng=rng,
        )

    def crossover(first, second, rng):
        swap_probability = config.genetic.crossover.params.get("swap_probability")
        return crossover_individuals(
            first,
            second,
            method=config.genetic.crossover.method,
            probability=config.genetic.crossover.probability,
            swap_probability=(
                float(swap_probability) if swap_probability is not None else None
            ),
            rng=rng,
        )

    allele_change = config.genetic.mutation.allele_change

    def mutate(individual, rng):
        return mutate_individual(
            individual,
            method=config.genetic.mutation.method,
            probability=config.genetic.mutation.probability,
            allele_mode=allele_change.mode,
            position_delta=allele_change.position_delta,
            color_delta=allele_change.color_delta,
            alpha_delta=allele_change.alpha_delta,
            alpha_range=config.representation.alpha_range,
            rng=rng,
        )

    def survive(population, offspring, generation, rng):
        selection = config.genetic.survival.selection

        def survival_selector(pool, count, current_generation, current_rng):
            return select_population(
                pool,
                count=count,
                method=selection.method,
                params=selection.params,
                generation=current_generation,
                fitness=_required_fitness,
                rng=current_rng,
            )

        return select_survivors(
            population,
            offspring,
            population_size=config.genetic.population_size,
            strategy=config.genetic.survival.strategy,
            generation=generation,
            selector=survival_selector,
            rng=rng,
        )

    stagnation = config.termination.stagnation

    def on_generation(metrics, population, best):
        del population
        every = config.output.checkpoint_every
        if every is None or metrics.generation == 0 or metrics.generation % every != 0:
            return
        checkpoint_image = render_individual(
            best,
            size=(
                original_target.size
                if config.output.render_original_size
                else working_target.size
            ),
            canvas_rgb=config.representation.canvas_rgb,
        )
        write_checkpoint(
            run_directory=run_directory,
            generation=metrics.generation,
            best=best,
            best_image=checkpoint_image,
        )

    evolution = evolve(
        initial_population,
        offspring_count=config.genetic.offspring_count,
        seed=config.run.seed,
        limits=EvolutionLimits(
            max_generations=config.termination.max_generations,
            target_error=config.termination.target_nmse,
            stagnation_patience=(stagnation.patience if stagnation else None),
            min_improvement=(stagnation.min_improvement if stagnation else 0.0),
            max_seconds=config.termination.max_seconds,
        ),
        evaluate=evaluator,
        error=_required_error,
        fitness=_required_fitness,
        select_parents=parent_selector,
        crossover=crossover,
        mutate=mutate,
        survive=survive,
        diversity=triangle_population_diversity,
        on_generation=on_generation,
    )

    output_size = (
        original_target.size
        if config.output.render_original_size
        else working_target.size
    )
    best_image = render_individual(
        evolution.best,
        size=output_size,
        canvas_rgb=config.representation.canvas_rgb,
    )
    sampled_metrics = []
    for metrics in evolution.metrics:
        if (
            metrics.generation % config.output.metrics_every != 0
            and metrics.generation != evolution.final_generation
        ):
            continue
        row = metrics.to_dict()
        row["stop_reason"] = (
            evolution.stop_reason
            if metrics.generation == evolution.final_generation
            else ""
        )
        sampled_metrics.append(row)
    metadata = {
        "mode": "evolution",
        "seed": config.run.seed,
        "stop_reason": evolution.stop_reason,
        "final_generation": evolution.final_generation,
        "best_generation": evolution.best_generation,
        "best_error": _required_error(evolution.best),
        "best_fitness": _required_fitness(evolution.best),
        "evaluations": evolution.evaluations,
        "population_size": config.genetic.population_size,
        "offspring_count": config.genetic.offspring_count,
        "triangle_count": config.representation.triangle_count,
        "working_size": list(working_target.size),
        "original_size": list(original_target.size),
        "output_size": list(output_size),
    }
    write_run_artifacts(
        run_directory=run_directory,
        effective_config=config.effective_dict(),
        metadata=metadata,
        metrics=sampled_metrics,
        best=evolution.best,
        best_image=best_image,
    )
    return ImageEvolutionResult(
        run_directory=run_directory,
        evolution=evolution,
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

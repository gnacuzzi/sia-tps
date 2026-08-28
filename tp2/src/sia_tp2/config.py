"""Strict loading and validation for the versioned JSON contract."""

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class ConfigError(ValueError):
    """All configuration problems discovered in one validation pass."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid configuration:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class InputConfig:
    image: Path
    working_max_side: int


@dataclass(frozen=True)
class RepresentationConfig:
    triangle_count: int
    canvas_rgb: Tuple[int, int, int]
    alpha_range: Tuple[int, int]
    initialization: str


@dataclass(frozen=True)
class SelectionConfig:
    method: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class CrossoverConfig:
    method: str
    probability: float
    params: Mapping[str, object]


@dataclass(frozen=True)
class AlleleChangeConfig:
    mode: str
    position_delta: Optional[float]
    color_delta: Optional[int]
    alpha_delta: Optional[int]


@dataclass(frozen=True)
class MutationConfig:
    method: str
    probability: float
    allele_change: AlleleChangeConfig


@dataclass(frozen=True)
class SurvivalConfig:
    strategy: str
    selection: SelectionConfig


@dataclass(frozen=True)
class GeneticConfig:
    population_size: int
    offspring_count: int
    parent_selection: SelectionConfig
    crossover: CrossoverConfig
    mutation: MutationConfig
    survival: SurvivalConfig


@dataclass(frozen=True)
class FitnessConfig:
    metric: str
    epsilon: float


@dataclass(frozen=True)
class StagnationConfig:
    patience: int
    min_improvement: float


@dataclass(frozen=True)
class TerminationConfig:
    max_generations: int
    target_nmse: Optional[float]
    stagnation: Optional[StagnationConfig]
    max_seconds: Optional[float]


@dataclass(frozen=True)
class RunConfig:
    seed: int


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    metrics_every: int
    checkpoint_every: Optional[int]
    render_original_size: bool


@dataclass(frozen=True)
class AppConfig:
    config_version: int
    input: InputConfig
    representation: RepresentationConfig
    genetic: GeneticConfig
    fitness: FitnessConfig
    termination: TerminationConfig
    run: RunConfig
    output: OutputConfig
    raw: Mapping[str, object]
    source_path: Path
    project_root: Path

    def effective_dict(self) -> dict:
        return deepcopy(dict(self.raw))


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: object, path: str, errors: List[str]) -> Dict[str, object]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set, path: str, errors: List[str]
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    for key in missing:
        errors.append(f"{path}.{key} is required")
    for key in unknown:
        errors.append(f"{path}.{key} is unknown")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(
    value: object,
    path: str,
    errors: List[str],
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if not _is_int(value):
        errors.append(f"{path} must be an integer")
        return 0
    if minimum is not None and value < minimum:
        errors.append(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        errors.append(f"{path} must be <= {maximum}")
    return value


def _number(
    value: object,
    path: str,
    errors: List[str],
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    exclusive_minimum: bool = False,
    exclusive_maximum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path} must be a number")
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        errors.append(f"{path} must be finite")
        return result
    if minimum is not None:
        invalid = result <= minimum if exclusive_minimum else result < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            errors.append(f"{path} must be {operator} {minimum}")
    if maximum is not None:
        invalid = result >= maximum if exclusive_maximum else result > maximum
        if invalid:
            operator = "<" if exclusive_maximum else "<="
            errors.append(f"{path} must be {operator} {maximum}")
    return result


def _string(value: object, path: str, errors: List[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")
        return ""
    return value


def _boolean(value: object, path: str, errors: List[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{path} must be a boolean")
        return False
    return value


def _choice(
    value: object, path: str, choices: set, errors: List[str]
) -> str:
    result = _string(value, path, errors)
    if result and result not in choices:
        errors.append(f"{path} must be one of {sorted(choices)}")
    return result


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validate_selection(
    value: object,
    path: str,
    errors: List[str],
    *,
    pool_size: int,
) -> SelectionConfig:
    obj = _object(value, path, errors)
    _exact_keys(obj, {"method", "params"}, path, errors)
    method = _choice(
        obj.get("method"),
        f"{path}.method",
        {
            "elite",
            "roulette",
            "universal",
            "ranking",
            "boltzmann",
            "tournament_deterministic",
            "tournament_probabilistic",
        },
        errors,
    )
    params = _object(obj.get("params"), f"{path}.params", errors)

    if method in {"elite", "roulette", "universal", "ranking"}:
        _exact_keys(params, set(), f"{path}.params", errors)
    elif method == "boltzmann":
        expected = {
            "initial_temperature",
            "final_temperature",
            "decay_rate",
        }
        _exact_keys(params, expected, f"{path}.params", errors)
        initial = _number(
            params.get("initial_temperature"),
            f"{path}.params.initial_temperature",
            errors,
            minimum=0.0,
            exclusive_minimum=True,
        )
        final = _number(
            params.get("final_temperature"),
            f"{path}.params.final_temperature",
            errors,
            minimum=0.0,
            exclusive_minimum=True,
        )
        _number(
            params.get("decay_rate"),
            f"{path}.params.decay_rate",
            errors,
            minimum=0.0,
        )
        if initial < final:
            errors.append(
                f"{path}.params.initial_temperature must be >= final_temperature"
            )
    elif method == "tournament_deterministic":
        _exact_keys(params, {"tournament_size"}, f"{path}.params", errors)
        size = _int(
            params.get("tournament_size"),
            f"{path}.params.tournament_size",
            errors,
            minimum=2,
        )
        if pool_size > 0 and size > pool_size:
            errors.append(
                f"{path}.params.tournament_size cannot exceed pool size {pool_size}"
            )
    elif method == "tournament_probabilistic":
        _exact_keys(params, {"threshold"}, f"{path}.params", errors)
        _number(
            params.get("threshold"),
            f"{path}.params.threshold",
            errors,
            minimum=0.5,
            maximum=1.0,
        )
    return SelectionConfig(method=method, params=deepcopy(params))


def _validate_config(raw: object, source_path: Path, project_root: Path) -> AppConfig:
    errors: List[str] = []
    root = _object(raw, "config", errors)
    _exact_keys(
        root,
        {
            "config_version",
            "input",
            "representation",
            "genetic",
            "fitness",
            "termination",
            "run",
            "output",
        },
        "config",
        errors,
    )

    config_version = _int(root.get("config_version"), "config_version", errors)
    if config_version != 1:
        errors.append("config_version must equal 1")

    input_obj = _object(root.get("input"), "input", errors)
    _exact_keys(input_obj, {"image", "working_max_side"}, "input", errors)
    image_value = _string(input_obj.get("image"), "input.image", errors)
    image_path = _resolve_path(image_value, project_root) if image_value else project_root
    if image_value:
        if not image_path.is_file():
            errors.append(f"input.image does not exist: {image_path}")
        elif image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            errors.append(
                "input.image must have a PNG, JPEG, or WebP extension"
            )
        else:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError) as error:
                errors.append(f"input.image cannot be decoded: {error}")
    working_max_side = _int(
        input_obj.get("working_max_side"),
        "input.working_max_side",
        errors,
        minimum=1,
    )

    rep_obj = _object(root.get("representation"), "representation", errors)
    _exact_keys(
        rep_obj,
        {"triangle_count", "canvas_rgb", "alpha_range", "initialization"},
        "representation",
        errors,
    )
    triangle_count = _int(
        rep_obj.get("triangle_count"),
        "representation.triangle_count",
        errors,
        minimum=1,
    )
    canvas_value = rep_obj.get("canvas_rgb")
    canvas_rgb = (0, 0, 0)
    if not isinstance(canvas_value, list) or len(canvas_value) != 3:
        errors.append("representation.canvas_rgb must contain exactly three integers")
    else:
        canvas_rgb = tuple(
            _int(channel, f"representation.canvas_rgb[{index}]", errors, minimum=0, maximum=255)
            for index, channel in enumerate(canvas_value)
        )
    alpha_value = rep_obj.get("alpha_range")
    alpha_range = (1, 255)
    if not isinstance(alpha_value, list) or len(alpha_value) != 2:
        errors.append("representation.alpha_range must contain [min, max]")
    else:
        alpha_range = tuple(
            _int(channel, f"representation.alpha_range[{index}]", errors, minimum=1, maximum=255)
            for index, channel in enumerate(alpha_value)
        )
        if alpha_range[0] > alpha_range[1]:
            errors.append("representation.alpha_range min must be <= max")
    initialization = _choice(
        rep_obj.get("initialization"),
        "representation.initialization",
        {"uniform_random"},
        errors,
    )

    genetic_obj = _object(root.get("genetic"), "genetic", errors)
    _exact_keys(
        genetic_obj,
        {
            "population_size",
            "offspring_count",
            "parent_selection",
            "crossover",
            "mutation",
            "survival",
        },
        "genetic",
        errors,
    )
    population_size = _int(
        genetic_obj.get("population_size"),
        "genetic.population_size",
        errors,
        minimum=2,
    )
    offspring_count = _int(
        genetic_obj.get("offspring_count"),
        "genetic.offspring_count",
        errors,
        minimum=1,
    )
    if offspring_count % 2 != 0:
        errors.append("genetic.offspring_count must be even")
    parent_selection = _validate_selection(
        genetic_obj.get("parent_selection"),
        "genetic.parent_selection",
        errors,
        pool_size=population_size,
    )

    crossover_obj = _object(
        genetic_obj.get("crossover"), "genetic.crossover", errors
    )
    _exact_keys(
        crossover_obj,
        {"method", "probability", "params"},
        "genetic.crossover",
        errors,
    )
    crossover_method = _choice(
        crossover_obj.get("method"),
        "genetic.crossover.method",
        {"one_point", "uniform"},
        errors,
    )
    crossover_probability = _number(
        crossover_obj.get("probability"),
        "genetic.crossover.probability",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    crossover_params = _object(
        crossover_obj.get("params"), "genetic.crossover.params", errors
    )
    if crossover_method == "one_point":
        _exact_keys(crossover_params, set(), "genetic.crossover.params", errors)
        if triangle_count < 2:
            errors.append("one_point crossover requires triangle_count >= 2")
    elif crossover_method == "uniform":
        _exact_keys(
            crossover_params,
            {"swap_probability"},
            "genetic.crossover.params",
            errors,
        )
        _number(
            crossover_params.get("swap_probability"),
            "genetic.crossover.params.swap_probability",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
    crossover = CrossoverConfig(
        method=crossover_method,
        probability=crossover_probability,
        params=deepcopy(crossover_params),
    )

    mutation_obj = _object(
        genetic_obj.get("mutation"), "genetic.mutation", errors
    )
    _exact_keys(
        mutation_obj,
        {"method", "probability", "allele_change"},
        "genetic.mutation",
        errors,
    )
    mutation_method = _choice(
        mutation_obj.get("method"),
        "genetic.mutation.method",
        {"single_gene", "multigene_uniform"},
        errors,
    )
    mutation_probability = _number(
        mutation_obj.get("probability"),
        "genetic.mutation.probability",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    allele_obj = _object(
        mutation_obj.get("allele_change"),
        "genetic.mutation.allele_change",
        errors,
    )
    allele_mode = _choice(
        allele_obj.get("mode"),
        "genetic.mutation.allele_change.mode",
        {"local_delta", "global_resample"},
        errors,
    )
    position_delta = None
    color_delta = None
    alpha_delta = None
    if allele_mode == "local_delta":
        _exact_keys(
            allele_obj,
            {"mode", "position_delta", "color_delta", "alpha_delta"},
            "genetic.mutation.allele_change",
            errors,
        )
        position_delta = _number(
            allele_obj.get("position_delta"),
            "genetic.mutation.allele_change.position_delta",
            errors,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        )
        color_delta = _int(
            allele_obj.get("color_delta"),
            "genetic.mutation.allele_change.color_delta",
            errors,
            minimum=1,
            maximum=255,
        )
        alpha_delta = _int(
            allele_obj.get("alpha_delta"),
            "genetic.mutation.allele_change.alpha_delta",
            errors,
            minimum=1,
            maximum=255,
        )
    elif allele_mode == "global_resample":
        _exact_keys(
            allele_obj,
            {"mode"},
            "genetic.mutation.allele_change",
            errors,
        )
    allele_change = AlleleChangeConfig(
        mode=allele_mode,
        position_delta=position_delta,
        color_delta=color_delta,
        alpha_delta=alpha_delta,
    )
    mutation = MutationConfig(
        method=mutation_method,
        probability=mutation_probability,
        allele_change=allele_change,
    )

    survival_obj = _object(
        genetic_obj.get("survival"), "genetic.survival", errors
    )
    _exact_keys(
        survival_obj, {"strategy", "selection"}, "genetic.survival", errors
    )
    survival_strategy = _choice(
        survival_obj.get("strategy"),
        "genetic.survival.strategy",
        {"additive", "exclusive"},
        errors,
    )
    if survival_strategy == "additive":
        survival_pool_size = population_size + offspring_count
    elif offspring_count > population_size:
        survival_pool_size = offspring_count
    else:
        survival_pool_size = population_size
    survival_selection = _validate_selection(
        survival_obj.get("selection"),
        "genetic.survival.selection",
        errors,
        pool_size=survival_pool_size,
    )
    survival = SurvivalConfig(
        strategy=survival_strategy, selection=survival_selection
    )

    fitness_obj = _object(root.get("fitness"), "fitness", errors)
    _exact_keys(fitness_obj, {"metric", "epsilon"}, "fitness", errors)
    fitness_metric = _choice(
        fitness_obj.get("metric"),
        "fitness.metric",
        {"normalized_mse"},
        errors,
    )
    epsilon = _number(
        fitness_obj.get("epsilon"),
        "fitness.epsilon",
        errors,
        minimum=0.0,
        maximum=1.0,
        exclusive_minimum=True,
        exclusive_maximum=True,
    )

    termination_obj = _object(root.get("termination"), "termination", errors)
    _exact_keys(
        termination_obj,
        {"max_generations", "target_nmse", "stagnation", "max_seconds"},
        "termination",
        errors,
    )
    max_generations = _int(
        termination_obj.get("max_generations"),
        "termination.max_generations",
        errors,
        minimum=1,
    )
    target_value = termination_obj.get("target_nmse")
    target_nmse = None
    if target_value is not None:
        target_nmse = _number(
            target_value,
            "termination.target_nmse",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
    stagnation_value = termination_obj.get("stagnation")
    stagnation = None
    if stagnation_value is not None:
        stagnation_obj = _object(
            stagnation_value, "termination.stagnation", errors
        )
        _exact_keys(
            stagnation_obj,
            {"patience", "min_improvement"},
            "termination.stagnation",
            errors,
        )
        stagnation = StagnationConfig(
            patience=_int(
                stagnation_obj.get("patience"),
                "termination.stagnation.patience",
                errors,
                minimum=1,
            ),
            min_improvement=_number(
                stagnation_obj.get("min_improvement"),
                "termination.stagnation.min_improvement",
                errors,
                minimum=0.0,
            ),
        )
    max_seconds_value = termination_obj.get("max_seconds")
    max_seconds = None
    if max_seconds_value is not None:
        max_seconds = _number(
            max_seconds_value,
            "termination.max_seconds",
            errors,
            minimum=0.0,
            exclusive_minimum=True,
        )

    run_obj = _object(root.get("run"), "run", errors)
    _exact_keys(run_obj, {"seed"}, "run", errors)
    seed = _int(run_obj.get("seed"), "run.seed", errors, minimum=0)

    output_obj = _object(root.get("output"), "output", errors)
    _exact_keys(
        output_obj,
        {
            "directory",
            "metrics_every",
            "checkpoint_every",
            "render_original_size",
        },
        "output",
        errors,
    )
    output_value = _string(output_obj.get("directory"), "output.directory", errors)
    output_path = _resolve_path(output_value, project_root) if output_value else project_root
    if output_path == image_path:
        errors.append("output.directory cannot equal input.image")
    if output_path.exists() and not output_path.is_dir():
        errors.append("output.directory cannot point to an existing file")
    metrics_every = _int(
        output_obj.get("metrics_every"),
        "output.metrics_every",
        errors,
        minimum=1,
    )
    checkpoint_value = output_obj.get("checkpoint_every")
    checkpoint_every = None
    if checkpoint_value is not None:
        checkpoint_every = _int(
            checkpoint_value,
            "output.checkpoint_every",
            errors,
            minimum=1,
        )
    render_original_size = _boolean(
        output_obj.get("render_original_size"),
        "output.render_original_size",
        errors,
    )

    if errors:
        raise ConfigError(errors)

    return AppConfig(
        config_version=config_version,
        input=InputConfig(image=image_path, working_max_side=working_max_side),
        representation=RepresentationConfig(
            triangle_count=triangle_count,
            canvas_rgb=canvas_rgb,
            alpha_range=alpha_range,
            initialization=initialization,
        ),
        genetic=GeneticConfig(
            population_size=population_size,
            offspring_count=offspring_count,
            parent_selection=parent_selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
        ),
        fitness=FitnessConfig(metric=fitness_metric, epsilon=epsilon),
        termination=TerminationConfig(
            max_generations=max_generations,
            target_nmse=target_nmse,
            stagnation=stagnation,
            max_seconds=max_seconds,
        ),
        run=RunConfig(seed=seed),
        output=OutputConfig(
            directory=output_path,
            metrics_every=metrics_every,
            checkpoint_every=checkpoint_every,
            render_original_size=render_original_size,
        ),
        raw=deepcopy(root),
        source_path=source_path,
        project_root=project_root,
    )


def load_config(
    path: Path, *, project_root: Optional[Path] = None
) -> AppConfig:
    source_path = Path(path).resolve()
    root = (project_root or PROJECT_ROOT).resolve()
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError([f"cannot read configuration {source_path}: {error}"]) from error
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ConfigError([f"invalid JSON in {source_path}: {error}"]) from error
    return _validate_config(raw, source_path, root)


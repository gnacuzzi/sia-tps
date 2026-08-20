"""Validated loading of the JSON execution configuration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .heuristics import HEURISTIC_NAMES
from .search.model import SearchLimits


class ConfigError(ValueError):
    """Raised when config.json is missing or contains invalid values."""


@dataclass(frozen=True)
class AppConfig:
    level_file: Path
    algorithm: str
    heuristic: Optional[str]
    cost_model: str
    limits: SearchLimits
    seed: int


def load_config(path: Union[str, Path]) -> AppConfig:
    """Load config.json and resolve its level path relative to that file."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Could not read config file: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in config file: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError("The configuration root must be a JSON object")

    level_value = _required_string(raw, "level_file")
    level_file = Path(level_value)
    if not level_file.is_absolute():
        level_file = config_path.parent / level_file

    algorithm = _required_string(raw, "algorithm").lower()
    if algorithm not in {"bfs", "dfs", "greedy", "astar"}:
        raise ConfigError(
            "algorithm must be one of: bfs, dfs, greedy, astar"
        )

    heuristic = raw.get("heuristic")
    if heuristic is not None and not isinstance(heuristic, str):
        raise ConfigError("heuristic must be a string or null")
    _validate_algorithm_heuristic(algorithm, heuristic)

    cost_model = _required_string(raw, "cost_model")
    limits = _load_limits(raw.get("limits"))

    seed = raw.get("seed")
    if type(seed) is not int:
        raise ConfigError("seed must be an integer")

    return AppConfig(
        level_file=level_file,
        algorithm=algorithm,
        heuristic=heuristic,
        cost_model=cost_model,
        limits=limits,
        seed=seed,
    )


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _validate_algorithm_heuristic(
    algorithm: str,
    heuristic: Optional[str],
) -> None:
    if algorithm in {"bfs", "dfs"}:
        if heuristic is not None:
            raise ConfigError(f"{algorithm} requires heuristic to be null")
        return

    if heuristic is None:
        raise ConfigError(f"{algorithm} requires a heuristic")
    if heuristic not in HEURISTIC_NAMES:
        allowed = ", ".join(sorted(HEURISTIC_NAMES))
        raise ConfigError(
            f"Unknown heuristic {heuristic!r}; expected one of: {allowed}"
        )


def _load_limits(raw: Any) -> SearchLimits:
    if not isinstance(raw, dict):
        raise ConfigError("limits must be a JSON object")

    max_expanded_nodes = raw.get("max_expanded_nodes")
    if max_expanded_nodes is not None and (
        type(max_expanded_nodes) is not int or max_expanded_nodes <= 0
    ):
        raise ConfigError("max_expanded_nodes must be a positive integer or null")

    timeout_seconds = raw.get("timeout_seconds")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ConfigError("timeout_seconds must be a positive number or null")

    try:
        return SearchLimits(
            max_expanded_nodes=max_expanded_nodes,
            timeout_seconds=(
                float(timeout_seconds) if timeout_seconds is not None else None
            ),
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error

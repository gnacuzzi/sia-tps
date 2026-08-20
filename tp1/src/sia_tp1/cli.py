"""Command-line entry point for loading and manually replaying a level."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from .config import ConfigError, load_config
from .domain import apply_move, is_goal
from .heuristics import get_heuristic
from .model import Direction
from .parser import LevelFormatError, parse_level
from .render import render_state
from .search import (
    SearchStatus,
    a_star_search,
    breadth_first_search,
    depth_first_search,
    greedy_search,
)
from .visualization import save_solution_gif


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.search:
            if arguments.moves:
                raise ConfigError("--search cannot be combined with --moves")
            return run_search(arguments.config, gif_path=arguments.gif)
        if arguments.gif is not None:
            raise ConfigError("--gif requires --search")
        return run(arguments.config, arguments.moves)
    except (ConfigError, LevelFormatError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


def run(
    config_path: Path,
    moves: Sequence[str],
    output: Optional[TextIO] = None,
) -> int:
    """Load one level, replay valid moves, and print their transitions."""

    if output is None:
        output = sys.stdout

    config = load_config(config_path)
    if config.cost_model != "unit":
        raise ConfigError("The Phase 1 runner only supports cost_model='unit'")

    level, state = parse_level(config.level_file)
    valid_moves = 0
    pushes = 0

    print("Initial state:", file=output)
    print(render_state(level, state), file=output)

    for index, move_name in enumerate(moves, start=1):
        direction = _parse_direction(move_name)
        transition = apply_move(level, state, direction)

        if transition is None:
            print(
                f"Move {index}: {direction.name}, invalid",
                file=output,
            )
            continue

        state = transition.state
        valid_moves += 1
        pushes += int(transition.pushed)
        pushed = str(transition.pushed).lower()
        print(
            f"Move {index}: {direction.name}, pushed={pushed}",
            file=output,
        )
        print(render_state(level, state), file=output)

        if is_goal(level, state):
            break

    solved = is_goal(level, state)
    print(f"Solved: {'yes' if solved else 'no'}", file=output)
    print(f"Valid moves: {valid_moves}", file=output)
    print(f"Pushes: {pushes}", file=output)
    print(f"Cost: {valid_moves}", file=output)
    return 0


def run_search(
    config_path: Path,
    output: Optional[TextIO] = None,
    gif_path: Optional[Path] = None,
) -> int:
    """Execute the configured search algorithm and print its result."""

    if output is None:
        output = sys.stdout

    config = load_config(config_path)
    if config.cost_model != "unit":
        raise ConfigError("Search only supports cost_model='unit'")

    level, initial_state = parse_level(config.level_file)
    if config.algorithm == "bfs":
        result = breadth_first_search(level, initial_state, config.limits)
    elif config.algorithm == "dfs":
        result = depth_first_search(level, initial_state, config.limits)
    elif config.algorithm == "greedy":
        heuristic = get_heuristic(config.heuristic)
        result = greedy_search(
            level,
            initial_state,
            heuristic,
            config.limits,
        )
    elif config.algorithm == "astar":
        heuristic = get_heuristic(config.heuristic)
        result = a_star_search(
            level,
            initial_state,
            heuristic,
            config.limits,
        )
    else:
        raise ConfigError(
            f"Algorithm {config.algorithm!r} is not implemented yet"
        )

    print(f"Status: {result.status.value}", file=output)
    print(f"Expanded nodes: {result.expanded_nodes}", file=output)
    print(f"Frontier at end: {result.frontier_size_at_end}", file=output)
    print(f"Maximum frontier: {result.max_frontier_size}", file=output)
    print(f"Elapsed seconds: {result.elapsed_seconds:.6f}", file=output)

    if result.status is SearchStatus.CUTOFF:
        print(f"Cutoff reason: {result.cutoff_reason.value}", file=output)
        _print_missing_gif(output, gif_path, result.status)
        return 0

    if result.status is SearchStatus.FAILURE:
        _print_missing_gif(output, gif_path, result.status)
        return 0

    print(f"Solution cost: {result.solution_cost:g}", file=output)
    print(f"Solution moves: {result.solution_moves}", file=output)
    print(f"Solution pushes: {result.solution_pushes}", file=output)
    print("Solution:", file=output)

    transitions = result.solution_transitions
    if transitions is not None:
        for index, transition in enumerate(transitions, start=1):
            pushed = str(transition.pushed).lower()
            print(
                f"{index}: {transition.direction.name}, pushed={pushed}",
                file=output,
            )

    print("Final state:", file=output)
    print(render_state(level, result.goal_node.state), file=output)

    if gif_path is not None:
        frame_count = save_solution_gif(
            level,
            result,
            config.algorithm,
            gif_path,
        )
        print(
            f"GIF saved to: {gif_path} ({frame_count} frames)",
            file=output,
        )
    return 0


def _print_missing_gif(
    output: TextIO,
    gif_path: Optional[Path],
    status: SearchStatus,
) -> None:
    if gif_path is not None:
        print(
            f"GIF not generated: search status is {status.value}",
            file=output,
        )


def _parse_direction(value: str) -> Direction:
    try:
        return Direction[value.upper()]
    except KeyError as error:
        allowed = ", ".join(direction.name for direction in Direction)
        raise ConfigError(
            f"Unknown direction {value!r}; expected one of: {allowed}"
        ) from error


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a Sokoban level and replay a manual move sequence."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--moves",
        nargs="*",
        default=[],
        metavar="DIRECTION",
        help="Manual sequence using UP, DOWN, LEFT, RIGHT",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run the search algorithm selected in config.json",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        metavar="PATH",
        help="Save the successful solution path as an animated GIF",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

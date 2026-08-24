"""Run reproducible Sokoban search experiments and export raw/summary CSVs."""

import argparse
import csv
import multiprocessing
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sia_tp1.heuristics import Heuristic, get_heuristic  # noqa: E402
from sia_tp1.model import Level, State  # noqa: E402
from sia_tp1.parser import parse_level  # noqa: E402
from sia_tp1.search import (  # noqa: E402
    SearchLimits,
    a_star_search,
    breadth_first_search,
    depth_first_search,
    greedy_search,
)


Case = Tuple[str, Optional[str]]
Metrics = Dict[str, object]
CaseRunner = Callable[[Path, str, Optional[str], SearchLimits], Metrics]

CORE_CASES: Tuple[Case, ...] = (
    ("bfs", None),
    ("dfs", None),
    ("greedy", "minimum_matching_manhattan"),
    ("greedy", "shortest_push_access"),
    ("astar", "minimum_matching_manhattan"),
    ("astar", "shortest_push_access"),
)

EXTENSION_CASES: Tuple[Case, ...] = (
    ("greedy", "deadlock_aware_reverse_push_matching"),
    ("greedy", "pair_pattern_database_matching"),
    ("astar", "deadlock_aware_reverse_push_matching"),
    ("astar", "pair_pattern_database_matching"),
)

RAW_FIELDS = (
    "run_order",
    "timestamp",
    "level",
    "algorithm",
    "heuristic",
    "repetition",
    "status",
    "cutoff_reason",
    "solution_cost",
    "solution_moves",
    "solution_pushes",
    "expanded_nodes",
    "frontier_size_at_end",
    "max_frontier_size",
    "elapsed_seconds",
    "initial_heuristic",
    "heuristic_evaluations",
    "heuristic_elapsed_seconds",
    "heuristic_seconds_per_evaluation",
    "solution",
    "max_expanded_nodes",
    "timeout_seconds",
    "seed",
)

SUMMARY_FIELDS = (
    "level",
    "algorithm",
    "heuristic",
    "runs",
    "successes",
    "failures",
    "cutoffs",
    "cutoff_reasons",
    "solution_cost_min",
    "solution_cost_median",
    "solution_moves_median",
    "solution_pushes_median",
    "expanded_nodes_median",
    "frontier_size_at_end_median",
    "max_frontier_size_median",
    "elapsed_seconds_median",
    "elapsed_seconds_min",
    "elapsed_seconds_max",
    "initial_heuristic_median",
    "heuristic_evaluations_median",
    "heuristic_elapsed_seconds_median",
    "heuristic_seconds_per_evaluation_median",
)


class _MeasuredHeuristic:
    """Measure heuristic values and evaluation time without changing results."""

    def __init__(self, heuristic: Heuristic) -> None:
        self._heuristic = heuristic
        self.initial_value: Optional[float] = None
        self.evaluations = 0
        self.elapsed_seconds = 0.0

    def __call__(self, level: Level, state: State) -> float:
        started_at = perf_counter()
        value = self._heuristic(level, state)
        self.elapsed_seconds += perf_counter() - started_at
        self.evaluations += 1
        if self.initial_value is None:
            self.initial_value = value
        return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    level_paths = tuple(path.resolve() for path in arguments.levels)
    cases = CORE_CASES + EXTENSION_CASES if arguments.suite == "all" else CORE_CASES
    limits = SearchLimits(
        max_expanded_nodes=arguments.max_expanded_nodes,
        timeout_seconds=arguments.timeout_seconds,
    )
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else _default_output_path()
    )
    summary_path = _summary_path(output_path)

    try:
        _validate_inputs(level_paths, output_path, summary_path, arguments.overwrite)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="") as raw_file:
            writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
            writer.writeheader()

            def save_row(row: Mapping[str, object]) -> None:
                writer.writerow(row)
                raw_file.flush()

            rows = collect_experiment_rows(
                level_paths=level_paths,
                cases=cases,
                repetitions=arguments.repetitions,
                limits=limits,
                seed=arguments.seed,
                on_row=save_row,
            )

        write_summary_csv(rows, summary_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"\nRaw results: {output_path}")
    print(f"Summary:     {summary_path}")
    return 0


def collect_experiment_rows(
    *,
    level_paths: Sequence[Path],
    cases: Sequence[Case],
    repetitions: int,
    limits: SearchLimits,
    seed: int,
    case_runner: Optional[CaseRunner] = None,
    on_row: Optional[Callable[[Mapping[str, object]], None]] = None,
) -> List[Metrics]:
    """Execute shuffled jobs and return one raw row per individual run."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if case_runner is None:
        case_runner = run_case_in_fresh_process

    jobs = [
        (level_path, algorithm, heuristic, repetition)
        for level_path in level_paths
        for algorithm, heuristic in cases
        for repetition in range(1, repetitions + 1)
    ]
    random.Random(seed).shuffle(jobs)

    rows: List[Metrics] = []
    total_jobs = len(jobs)
    for run_order, job in enumerate(jobs, start=1):
        level_path, algorithm, heuristic, repetition = job
        heuristic_label = heuristic if heuristic is not None else "null"
        print(
            f"[{run_order}/{total_jobs}] {level_path.name} | {algorithm} | "
            f"{heuristic_label} | repetition {repetition}",
            flush=True,
        )

        metrics = case_runner(level_path, algorithm, heuristic, limits)
        row: Metrics = {
            "run_order": run_order,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": _display_path(level_path),
            "algorithm": algorithm,
            "heuristic": heuristic_label,
            "repetition": repetition,
            **metrics,
            "max_expanded_nodes": limits.max_expanded_nodes,
            "timeout_seconds": limits.timeout_seconds,
            "seed": seed,
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)

    return rows


def run_case_in_fresh_process(
    level_path: Path,
    algorithm: str,
    heuristic: Optional[str],
    limits: SearchLimits,
) -> Metrics:
    """Run one search in a spawned interpreter so heuristic caches start cold."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(
            sender,
            str(level_path),
            algorithm,
            heuristic,
            limits.max_expanded_nodes,
            limits.timeout_seconds,
        ),
    )
    process.start()
    sender.close()

    try:
        payload = receiver.recv()
    except EOFError as error:
        process.join()
        raise RuntimeError(
            f"Experiment worker exited without a result (code {process.exitcode})"
        ) from error
    finally:
        receiver.close()

    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"Experiment worker exited with code {process.exitcode}"
        )
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload["metrics"]


def _worker_entry(
    connection,
    level_path: str,
    algorithm: str,
    heuristic: Optional[str],
    max_expanded_nodes: Optional[int],
    timeout_seconds: Optional[float],
) -> None:
    try:
        limits = SearchLimits(max_expanded_nodes, timeout_seconds)
        metrics = _run_case(Path(level_path), algorithm, heuristic, limits)
        connection.send({"metrics": metrics})
    except Exception as error:
        connection.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def _run_case(
    level_path: Path,
    algorithm: str,
    heuristic_name: Optional[str],
    limits: SearchLimits,
) -> Metrics:
    level, initial_state = parse_level(level_path)
    measured_heuristic: Optional[_MeasuredHeuristic] = None
    if algorithm == "bfs":
        result = breadth_first_search(level, initial_state, limits)
    elif algorithm == "dfs":
        result = depth_first_search(level, initial_state, limits)
    elif algorithm in {"greedy", "astar"}:
        if heuristic_name is None:
            raise ValueError(f"{algorithm} requires a heuristic")
        measured_heuristic = _MeasuredHeuristic(get_heuristic(heuristic_name))
        if algorithm == "greedy":
            result = greedy_search(
                level,
                initial_state,
                measured_heuristic,
                limits,
            )
        else:
            result = a_star_search(
                level,
                initial_state,
                measured_heuristic,
                limits,
            )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    transitions = result.solution_transitions
    solution = (
        " ".join(transition.direction.name for transition in transitions)
        if transitions is not None
        else ""
    )
    heuristic_evaluations = (
        measured_heuristic.evaluations if measured_heuristic is not None else 0
    )
    heuristic_elapsed_seconds = (
        measured_heuristic.elapsed_seconds
        if measured_heuristic is not None
        else 0.0
    )
    return {
        "status": result.status.value,
        "cutoff_reason": (
            result.cutoff_reason.value if result.cutoff_reason is not None else ""
        ),
        "solution_cost": result.solution_cost,
        "solution_moves": result.solution_moves,
        "solution_pushes": result.solution_pushes,
        "expanded_nodes": result.expanded_nodes,
        "frontier_size_at_end": result.frontier_size_at_end,
        "max_frontier_size": result.max_frontier_size,
        "elapsed_seconds": round(result.elapsed_seconds, 9),
        "initial_heuristic": (
            measured_heuristic.initial_value
            if measured_heuristic is not None
            else None
        ),
        "heuristic_evaluations": heuristic_evaluations,
        "heuristic_elapsed_seconds": round(heuristic_elapsed_seconds, 9),
        "heuristic_seconds_per_evaluation": (
            round(heuristic_elapsed_seconds / heuristic_evaluations, 12)
            if heuristic_evaluations
            else None
        ),
        "solution": solution,
    }


def write_summary_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Aggregate raw repetitions by level, algorithm, and heuristic."""

    grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["level"]), str(row["algorithm"]), str(row["heuristic"]))
        grouped.setdefault(key, []).append(row)

    with path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for key in sorted(grouped):
            level, algorithm, heuristic = key
            group = grouped[key]
            successful = [row for row in group if row["status"] == "success"]
            elapsed = [float(row["elapsed_seconds"]) for row in group]
            writer.writerow(
                {
                    "level": level,
                    "algorithm": algorithm,
                    "heuristic": heuristic,
                    "runs": len(group),
                    "successes": len(successful),
                    "failures": sum(row["status"] == "failure" for row in group),
                    "cutoffs": sum(row["status"] == "cutoff" for row in group),
                    "cutoff_reasons": "|".join(
                        sorted(
                            {
                                str(row["cutoff_reason"])
                                for row in group
                                if row["cutoff_reason"]
                            }
                        )
                    ),
                    "solution_cost_min": _minimum(successful, "solution_cost"),
                    "solution_cost_median": _median(successful, "solution_cost"),
                    "solution_moves_median": _median(successful, "solution_moves"),
                    "solution_pushes_median": _median(successful, "solution_pushes"),
                    "expanded_nodes_median": _median(group, "expanded_nodes"),
                    "frontier_size_at_end_median": _median(
                        group, "frontier_size_at_end"
                    ),
                    "max_frontier_size_median": _median(
                        group, "max_frontier_size"
                    ),
                    "elapsed_seconds_median": round(statistics.median(elapsed), 9),
                    "elapsed_seconds_min": round(min(elapsed), 9),
                    "elapsed_seconds_max": round(max(elapsed), 9),
                    "initial_heuristic_median": _median(
                        group, "initial_heuristic"
                    ),
                    "heuristic_evaluations_median": _median(
                        group, "heuristic_evaluations"
                    ),
                    "heuristic_elapsed_seconds_median": _median(
                        group, "heuristic_elapsed_seconds"
                    ),
                    "heuristic_seconds_per_evaluation_median": _median(
                        group, "heuristic_seconds_per_evaluation"
                    ),
                }
            )


def _median(rows: Sequence[Mapping[str, object]], field: str):
    values = [float(row[field]) for row in rows if row[field] is not None]
    return statistics.median(values) if values else ""


def _minimum(rows: Sequence[Mapping[str, object]], field: str):
    values = [float(row[field]) for row in rows if row[field] is not None]
    return min(values) if values else ""


def _validate_inputs(
    level_paths: Sequence[Path],
    output_path: Path,
    summary_path: Path,
    overwrite: bool,
) -> None:
    for level_path in level_paths:
        if not level_path.is_file():
            raise ValueError(f"Level file does not exist: {level_path}")
        parse_level(level_path)
    if not overwrite:
        for path in (output_path, summary_path):
            if path.exists():
                raise ValueError(
                    f"Output already exists: {path}; use --overwrite to replace it"
                )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "results" / f"experiments_{timestamp}.csv"


def _summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated Sokoban search experiments and export CSV files."
    )
    parser.add_argument("levels", type=Path, nargs="+", help="Level files to test")
    parser.add_argument(
        "--suite",
        choices=("core", "all"),
        default="core",
        help="core runs 6 required cases; all includes 4 extension cases",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
        help="Runs per case (default: 1; use 10 for final timing analysis)",
    )
    parser.add_argument(
        "--max-expanded-nodes",
        type=_positive_int,
        default=1_000_000,
        help="Expansion cutoff per run (default: 1000000)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=25.0,
        help="Time cutoff per run (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used only to shuffle execution order (default: 0)",
    )
    parser.add_argument("--output", type=Path, help="Raw CSV destination")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the raw and summary CSV files",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

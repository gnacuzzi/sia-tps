"""Generate the six algorithm/heuristic GIF variants for one level."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("bfs", None, "bfs.gif"),
    ("dfs", None, "dfs.gif"),
    ("greedy", "minimum_matching_manhattan", "greedy_mmm.gif"),
    ("greedy", "shortest_push_access", "greedy_spa.gif"),
    ("astar", "shortest_push_access", "astar_spa.gif"),
    ("astar", "minimum_matching_manhattan", "astar_mmm.gif"),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate all six Sokoban solution GIF variants."
    )
    parser.add_argument("level", type=Path, help="Level file to solve")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.json",
        help="Configuration file to modify temporarily",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: output/videos/<level name>)",
    )
    arguments = parser.parse_args(argv)

    level_path = arguments.level.resolve()
    config_path = arguments.config.resolve()
    output_dir = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else PROJECT_ROOT / "output" / "videos" / level_path.stem
    )

    try:
        return generate_all_gifs(level_path, config_path, output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


def generate_all_gifs(
    level_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Run all cases and restore the exact original config in every outcome."""

    if not level_path.is_file():
        raise ValueError(f"Level file does not exist: {level_path}")
    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")

    original_config = config_path.read_bytes()
    base_config = json.loads(original_config.decode("utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("The configuration root must be a JSON object")

    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not current_python_path
        else source_path + os.pathsep + current_python_path
    )

    try:
        for algorithm, heuristic, filename in CASES:
            case_config = dict(base_config)
            case_config["level_file"] = str(level_path)
            case_config["algorithm"] = algorithm
            case_config["heuristic"] = heuristic
            config_path.write_text(
                json.dumps(case_config, indent=2) + "\n",
                encoding="utf-8",
            )

            gif_path = output_dir / filename
            heuristic_label = heuristic if heuristic is not None else "none"
            print(
                f"\nGenerating {filename}: "
                f"algorithm={algorithm}, heuristic={heuristic_label}",
                flush=True,
            )
            completed = runner(
                [
                    sys.executable,
                    "-m",
                    "sia_tp1",
                    "--config",
                    str(config_path),
                    "--search",
                    "--gif",
                    str(gif_path),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                print(
                    f"Stopped: {filename} exited with code "
                    f"{completed.returncode}.",
                    file=sys.stderr,
                )
                return completed.returncode
    finally:
        config_path.write_bytes(original_config)
        print(f"\nOriginal config restored: {config_path}", flush=True)

    print(f"All GIFs generated in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

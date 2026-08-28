"""Command-line interface for configuration inspection and Phase 1 smoke runs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from sia_tp2.config import ConfigError, load_config
from sia_tp2.workflow import render_random_population


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "inspect-config":
            print(
                json.dumps(
                    config.effective_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "render-random":
            result = render_random_population(config)
            print(f"Run directory: {result.run_directory}")
            print(f"Best individual index: {result.best_index}")
            print(f"NMSE: {result.best.error:.12f}")
            print(f"Fitness: {result.best.fitness:.12f}")
            return 0
    except (ConfigError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sia-tp2",
        description="Approximate images with a configurable genetic representation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the strict JSON configuration.",
    )
    parser.add_argument(
        "command",
        choices=("inspect-config", "render-random"),
        help="Operation to execute.",
    )
    return parser


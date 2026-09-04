#!/usr/bin/env python3
"""Materialize and execute a reproducible TP2 experiment phase."""

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from sia_tp2.study import (
    build_specs,
    build_showcase_specs,
    load_manifest,
    materialize_specs,
    read_records,
    run_specs,
    selection_profile,
    write_profile,
    write_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "study-manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-comparative-study"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the staged, reproducible TP2 comparative study."
    )
    parser.add_argument(
        "phase",
        choices=("profile", "selection", "crossover", "mutation", "validation", "showcase"),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selected",
        type=Path,
        action="append",
        help="JSON decisions from analyze_study.py; required after selection.",
    )
    parser.add_argument(
        "--validation-records",
        type=Path,
        help="Validation records used to select a median showcase seed; required for showcase.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write validated JSON configurations but do not evolve them.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Run every spec even if a matching raw result already exists.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.phase == "profile":
        profile = selection_profile(manifest)
        path = args.output / "profile" / "selection-pressure.csv"
        write_profile(profile, path)
        print(f"Selection profile: {path}")
        return 0

    selected = _load_selected(args.selected) if args.selected else None
    if args.phase == "showcase":
        if not args.validation_records:
            parser.error("showcase requires --validation-records")
        specs = build_showcase_specs(
            manifest,
            output_root=args.output,
            selected_methods=selected or {},
            validation_records=read_records(args.validation_records),
        )
    else:
        specs = build_specs(
            manifest,
            phase=args.phase,
            output_root=args.output,
            selected_methods=selected,
        )
    config_directory = args.output / "configs" / args.phase
    materialize_specs(specs, config_directory)
    print(f"Materialized {len(specs)} configs in {config_directory}")
    if args.prepare_only:
        return 0

    records = run_specs(
        specs,
        config_directory=config_directory,
        project_root=PROJECT_ROOT,
        resume=not args.no_resume,
        progress=print,
    )
    record_path = args.output / "records" / f"{args.phase}.csv"
    write_records(records, record_path)
    print(f"Completed {len(records)} runs: {record_path}")
    return 0


def _load_selected(paths: Sequence[Path]) -> Mapping[str, Sequence[str]]:
    selected = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("selected-methods JSON must be an object")
        selected.update(payload)
    return selected


if __name__ == "__main__":
    raise SystemExit(main())

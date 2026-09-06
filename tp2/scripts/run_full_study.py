#!/usr/bin/env python3
"""Run the complete TP2 study, carrying numerical decisions between phases."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "study-manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-colombia-final-study"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run, analyze and connect every phase of the final TP2 study."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-showcase",
        action="store_true",
        help="Stop after validation instead of running the three extended visual runs.",
    )
    args = parser.parse_args(argv)

    source_manifest = _read_object(args.manifest)
    _validate_output_identity(source_manifest, args.output)
    working_manifest = args.output / "working-manifest.json"
    published = args.output / "published"
    _write_object(source_manifest, working_manifest)

    _run_and_analyze("profile", working_manifest, args.output, published)

    _run_and_analyze("resolution", working_manifest, args.output, published)
    resolution = _decision(args.output, "resolution")
    _set_target_integer(source_manifest, "working_max_side", resolution, "side_")
    _write_object(source_manifest, working_manifest)

    _run_and_analyze("capacity", working_manifest, args.output, published)
    capacity = _decision(args.output, "capacity")
    _set_target_integer(source_manifest, "triangle_count", capacity, "triangles_")
    _write_object(source_manifest, working_manifest)

    _run_and_analyze("selection", working_manifest, args.output, published)
    selection_decision = args.output / "decisions" / "selection.json"

    _run_and_analyze(
        "crossover",
        working_manifest,
        args.output,
        published,
        selected=(selection_decision,),
    )
    crossover_decision = args.output / "decisions" / "crossover.json"
    _apply_base_decisions(source_manifest, crossover_decision)
    _write_object(source_manifest, working_manifest)

    _run_and_analyze(
        "mutation",
        working_manifest,
        args.output,
        published,
        selected=(selection_decision, crossover_decision),
    )
    mutation_decision = args.output / "decisions" / "mutation.json"
    _apply_base_decisions(source_manifest, mutation_decision)
    _write_object(source_manifest, working_manifest)

    _run_and_analyze(
        "survival",
        working_manifest,
        args.output,
        published,
        selected=(selection_decision, crossover_decision, mutation_decision),
    )
    survival_decision = args.output / "decisions" / "survival.json"
    _apply_base_decisions(source_manifest, survival_decision)
    _write_object(source_manifest, working_manifest)

    final_decisions = (
        selection_decision,
        crossover_decision,
        mutation_decision,
        survival_decision,
    )
    _run_and_analyze(
        "validation",
        working_manifest,
        args.output,
        published,
        selected=final_decisions,
    )

    if not args.no_showcase:
        _run_and_analyze(
            "showcase",
            working_manifest,
            args.output,
            published,
            selected=final_decisions,
            validation_records=args.output / "records" / "validation.csv",
        )

    _call(
        "generate_comparative_report.py",
        "--output",
        str(args.output),
        "--results",
        str(published),
    )
    print(f"\nComplete study: {args.output}")
    print(f"Final numerical decisions: {args.output / 'decisions'}")
    print(f"Report and selected artifacts: {published}")
    return 0


def _run_and_analyze(
    phase: str,
    manifest: Path,
    output: Path,
    published: Path,
    *,
    selected: Sequence[Path] = (),
    validation_records: Optional[Path] = None,
) -> None:
    run_arguments = [phase, "--manifest", str(manifest), "--output", str(output)]
    for decision in selected:
        run_arguments.extend(("--selected", str(decision)))
    if validation_records is not None:
        run_arguments.extend(("--validation-records", str(validation_records)))
    _call("run_study.py", *run_arguments)
    _call(
        "analyze_study.py",
        phase,
        "--output",
        str(output),
        "--published-output",
        str(published),
        "--figures",
    )


def _call(script: str, *arguments: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _decision(output: Path, phase: str) -> str:
    payload = _read_object(output / "decisions" / f"{phase}.json")
    values = payload.get(phase)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f"{phase} must produce exactly one textual decision")
    print(f"Automatic {phase} decision: {values[0]}")
    return values[0]


def _set_target_integer(
    manifest: MutableMapping[str, object], key: str, condition: str, prefix: str
) -> None:
    if not condition.startswith(prefix):
        raise ValueError(f"unexpected condition for {key}: {condition}")
    value = int(condition[len(prefix) :])
    targets = _mutable_object(manifest["targets"], "targets")
    flag = _mutable_object(targets["flag"], "targets.flag")
    flag[key] = value


def _apply_base_decisions(manifest: MutableMapping[str, object], path: Path) -> None:
    decisions = _read_object(path)
    base = _mutable_object(manifest["base"], "base")
    for key in ("selection", "crossover", "mutation", "survival"):
        values = decisions.get(key)
        if values is None:
            continue
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
            raise ValueError(f"{path} must contain one value for {key}")
        manifest_key = "selector" if key == "selection" else key
        base[manifest_key] = values[0]


def _validate_output_identity(manifest: Mapping[str, object], output: Path) -> None:
    snapshot = output / "source-manifest.json"
    if snapshot.is_file() and _read_object(snapshot) != manifest:
        raise ValueError(
            f"{output} contains runs for a different manifest; choose another --output"
        )
    if not snapshot.exists():
        _write_object(manifest, snapshot)


def _read_object(path: Path) -> MutableMapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_object(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mutable_object(value: object, name: str) -> MutableMapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

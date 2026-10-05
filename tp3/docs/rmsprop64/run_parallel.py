"""Ejecutar un plan de búsqueda con una corrida por proceso.

Cada corrida conserva su semilla, así que el resultado es idéntico al de
``run_experiment.py``; sólo cambia que varias corridas avanzan en paralelo. La
carpeta final mantiene la misma estructura: una subcarpeta por corrida y un
``summary.csv`` consolidado.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sia_tp3.digit_experiment import _expanded_runs, load_digit_search_config

RUNNER = Path(__file__).resolve().parents[1] / "ejercicio2" / "run_experiment.py"


def _run_one(run, config, args, scratch):
    single = {key: value for key, value in config.items()
              if key not in {"runs", "repeat_seeds"}}
    single["runs"] = [run]
    part = scratch / run["name"]
    config_path = scratch / f"{run['name']}.json"
    config_path.write_text(json.dumps(single, indent=2) + "\n")
    command = [sys.executable, str(RUNNER), "--config", str(config_path),
               "--data", str(args.data), "--output", str(part)]
    if args.additional_data:
        command += ["--additional-data", str(args.additional_data),
                    "--deduplicate-inputs"]
    log = (scratch / f"{run['name']}.log").open("w")
    status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if status != 0:
        raise RuntimeError(f"{run['name']} terminó con código {status}")
    return part


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--additional-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = load_digit_search_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        parser.exit(2, "Error: output debe ser una carpeta nueva o vacía\n")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    runs = _expanded_runs(config)
    scratch = Path(tempfile.mkdtemp(prefix="rmsprop64-"))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(lambda run: _run_one(run, config, args, scratch), runs))

    rows = []
    for run, part in zip(runs, parts):
        shutil.move(str(part / run["name"]), str(args.output / run["name"]))
        with (part / "summary.csv").open(newline="") as file:
            rows.extend(csv.DictReader(file))
        for name in ("split-indices.npz", "development-classes.json",
                     "data-source.json"):
            if not (args.output / name).exists():
                shutil.copy(part / name, args.output / name)
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(scratch)
    print(f"{len(rows)} corridas en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

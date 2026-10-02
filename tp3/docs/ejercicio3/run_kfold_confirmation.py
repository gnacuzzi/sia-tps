"""Ejecutar una confirmación k-fold congelada sin recibir un archivo de test."""

import argparse
import json
from pathlib import Path

from sia_tp3.digit_experiment import load_digit_search_config, run_digit_search


def run(plan_path, data, additional_data, output):
    plan = json.loads(Path(plan_path).read_text())
    if (set(plan) != {"protocol", "fold_count", "fold_seed", "paired_seeds",
                     "checkpoints", "run"}
            or plan["protocol"] != "kfold-confirmation"):
        raise ValueError("plan de confirmación k-fold inválido")
    fold_count = plan["fold_count"]
    seeds = plan["paired_seeds"]
    if (type(fold_count) is not int or fold_count < 2
            or seeds != list(range(fold_count))):
        raise ValueError("se requiere una semilla preasignada por fold")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for fold_index, seed in enumerate(seeds):
        candidate = dict(plan["run"])
        candidate.update(
            name=f"frozen-candidate-fold-{fold_index}",
            model_seed=seed,
            shuffle_seed=seed,
            augmentation=dict(candidate["augmentation"], seed=seed),
        )
        config = {
            "protocol": "search",
            "fold_count": fold_count,
            "fold_index": fold_index,
            "fold_seed": plan["fold_seed"],
            "checkpoints": plan["checkpoints"],
            "runs": [candidate],
        }
        config_path = output / f"fold-{fold_index}-config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        validated = load_digit_search_config(config_path)
        run_digit_search(
            validated, data, output / f"fold-{fold_index}",
            additional_data_path=additional_data, deduplicate_inputs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--additional-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.plan, args.data, args.additional_data, args.output)

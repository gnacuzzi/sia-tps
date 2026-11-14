"""Embudo reproducible de capacidad del AE para funciones Python one-hot."""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .autoencoder import Autoencoder
from .code_dataset import (SOURCE_LENGTH, VOCABULARY_SIZE, belongs_to_grammar,
                           compiles, decode_patterns, function_parameters,
                           load_code_dataset)
from .optimizers import optimizer_from_config


def _write_csv(path, rows, fieldnames=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames es obligatorio cuando rows esta vacio")
        fieldnames = rows[0].keys()
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def categorical_source_metrics(model, patterns, expected_sources):
    """Medir reconstrucción por carácter y por fuente tras decodificar argmax."""
    probabilities = model.predict(patterns)
    grouped = probabilities.reshape(
        len(patterns), SOURCE_LENGTH, VOCABULARY_SIZE
    )
    predicted_indices = np.argmax(grouped, axis=2)
    expected_indices = np.argmax(
        patterns.reshape(len(patterns), SOURCE_LENGTH, VOCABULARY_SIZE), axis=2
    )
    correct = predicted_indices == expected_indices
    decoded = decode_patterns(probabilities)
    exact = np.all(correct, axis=1)
    compilable = np.asarray([compiles(source) for source in decoded])
    grammar = np.asarray([belongs_to_grammar(source) for source in decoded])
    loss = model.loss(patterns)
    return {
        "loss": loss,
        "cross_entropy_per_character": loss / SOURCE_LENGTH,
        "perplexity_per_character": float(np.exp(loss / SOURCE_LENGTH)),
        "exact_sources": int(exact.sum()),
        "exact_source_rate": float(exact.mean()),
        "character_accuracy": float(correct.mean()),
        "mean_character_errors": float((~correct).sum(axis=1).mean()),
        "max_character_errors": int((~correct).sum(axis=1).max()),
        "compile_rate": float(compilable.mean()),
        "grammar_rate": float(grammar.mean()),
        "mean_selected_probability": float(
            np.take_along_axis(grouped, expected_indices[..., None], axis=2).mean()
        ),
        "decoded": decoded,
        "correct_by_position": correct.mean(axis=0),
        "exact_mask": exact,
    }


def _public_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if key not in {"decoded", "correct_by_position", "exact_mask"}
    }


def _parameter_count(model):
    return int(sum(parameter.size for parameter in model.parameters))


def _train_candidate(configuration, candidate, dataset, output):
    model = Autoencoder(
        candidate["architecture"],
        latent_layer=configuration["latent_layer"],
        activations=configuration["activations"],
        loss="categorical_cross_entropy",
        categorical_groups=SOURCE_LENGTH,
        seed=configuration["model_seed"],
        init_scale=configuration["init_scale"],
    )
    optimizer = optimizer_from_config(configuration["optimizer"])
    history = []
    consecutive_gate_hits = 0
    started = time.perf_counter()
    for epoch in range(configuration["max_epochs"] + 1):
        should_evaluate = (
            epoch == 0
            or epoch % configuration["evaluation_interval"] == 0
            or epoch == configuration["max_epochs"]
        )
        if should_evaluate:
            train = categorical_source_metrics(
                model, dataset.train_patterns, dataset.train_sources
            )
            validation = categorical_source_metrics(
                model, dataset.validation_patterns, dataset.validation_sources
            )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value
                   for key, value in _public_metrics(train).items()},
                **{f"validation_{key}": value
                   for key, value in _public_metrics(validation).items()},
            }
            history.append(row)
            gate_hit = (
                train["exact_source_rate"]
                >= configuration["selection_gates"]["train_exact_source_rate"]
                and validation["exact_source_rate"]
                >= configuration["selection_gates"]["validation_exact_source_rate"]
                and validation["grammar_rate"]
                >= configuration["selection_gates"]["validation_grammar_rate"]
            )
            consecutive_gate_hits = consecutive_gate_hits + 1 if gate_hit else 0
            print(json.dumps({
                "candidate": candidate["name"],
                "epoch": epoch,
                "train_exact": train["exact_source_rate"],
                "validation_exact": validation["exact_source_rate"],
                "validation_grammar": validation["grammar_rate"],
            }), flush=True)
            if consecutive_gate_hits >= configuration["required_gate_hits"]:
                break
        if epoch == configuration["max_epochs"]:
            break
        weight_gradients, bias_gradients = model.gradients(
            dataset.train_patterns
        )
        optimizer.step(model.parameters, weight_gradients + bias_gradients)

    elapsed = time.perf_counter() - started
    train = categorical_source_metrics(
        model, dataset.train_patterns, dataset.train_sources
    )
    validation = categorical_source_metrics(
        model, dataset.validation_patterns, dataset.validation_sources
    )
    passed = (
        train["exact_source_rate"]
        >= configuration["selection_gates"]["train_exact_source_rate"]
        and validation["exact_source_rate"]
        >= configuration["selection_gates"]["validation_exact_source_rate"]
        and validation["grammar_rate"]
        >= configuration["selection_gates"]["validation_grammar_rate"]
    )
    candidate_output = output / candidate["name"]
    candidate_output.mkdir()
    model.save(candidate_output / "model.npz")
    _write_csv(candidate_output / "history.csv", history)
    summary = {
        "name": candidate["name"],
        "architecture": "-".join(map(str, candidate["architecture"])),
        "parameter_count": _parameter_count(model),
        "epochs_completed": history[-1]["epoch"],
        "training_seconds": elapsed,
        "passed_selection_gates": passed,
        **{f"train_{key}": value
           for key, value in _public_metrics(train).items()},
        **{f"validation_{key}": value
           for key, value in _public_metrics(validation).items()},
    }
    (candidate_output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return model, summary


def _diagnostic_rows(model, dataset):
    position_rows = []
    operator_pair_rows = []
    failure_rows = []
    for split, patterns, expected_sources in (
            ("train", dataset.train_patterns, dataset.train_sources),
            ("validation", dataset.validation_patterns,
             dataset.validation_sources),
            ("test", dataset.test_patterns, dataset.test_sources)):
        metrics = categorical_source_metrics(model, patterns, expected_sources)
        for position, accuracy in enumerate(metrics["correct_by_position"]):
            position_rows.append({
                "split": split,
                "position": position,
                "accuracy": accuracy,
            })
        buckets = {}
        for index, expected in enumerate(expected_sources):
            left_operator, _, right_operator, _ = function_parameters(expected)
            bucket = buckets.setdefault((left_operator, right_operator), [])
            bucket.append(index)
            if not metrics["exact_mask"][index]:
                failure_rows.append({
                    "split": split,
                    "expected": expected,
                    "decoded": metrics["decoded"][index],
                    "character_errors": int(np.sum(
                        np.asarray(list(expected))
                        != np.asarray(list(metrics["decoded"][index]))
                    )),
                    "compiles": compiles(metrics["decoded"][index]),
                    "belongs_to_grammar": belongs_to_grammar(
                        metrics["decoded"][index]
                    ),
                })
        for (left_operator, right_operator), indices in buckets.items():
            exact_rate = float(metrics["exact_mask"][indices].mean())
            operator_pair_rows.append({
                "split": split,
                "left_operator": left_operator,
                "right_operator": right_operator,
                "samples": len(indices),
                "exact_source_rate": exact_rate,
            })
    return position_rows, operator_pair_rows, failure_rows


def run(configuration):
    dataset = load_code_dataset(split_seed=configuration["split_seed"])
    output = Path(configuration["result_root"])
    if output.exists():
        raise FileExistsError(f"el estudio ya existe: {output}")
    output.mkdir(parents=True)
    summaries = []
    selected = None
    for candidate in configuration["candidates"]:
        model, summary = _train_candidate(
            configuration, candidate, dataset, output
        )
        summaries.append(summary)
        if summary["passed_selection_gates"]:
            selected = (candidate, model, summary)
            break
    _write_csv(output / "comparison.csv", summaries)
    if selected is None:
        result = {
            "selected_candidate": None,
            "reason": "ninguna arquitectura alcanzo los umbrales prefijados",
            "candidates_run": len(summaries),
        }
    else:
        candidate, model, summary = selected
        test = categorical_source_metrics(
            model, dataset.test_patterns, dataset.test_sources
        )
        position_rows, pair_rows, failure_rows = _diagnostic_rows(model, dataset)
        _write_csv(output / "selected-position-accuracy.csv", position_rows)
        _write_csv(output / "selected-operator-pairs.csv", pair_rows)
        _write_csv(
            output / "selected-failures.csv", failure_rows,
            fieldnames=("split", "expected", "decoded", "character_errors",
                        "compiles", "belongs_to_grammar"),
        )
        result = {
            "selected_candidate": candidate["name"],
            "selection_rule": "primera arquitectura que supera los umbrales",
            "candidates_run": len(summaries),
            **summary,
            **{f"test_{key}": value
               for key, value in _public_metrics(test).items()},
        }
        shutil.copyfile(
            output / candidate["name"] / "model.npz",
            output / "selected-model.npz",
        )
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    print(json.dumps(run(configuration), indent=2))


if __name__ == "__main__":
    main()

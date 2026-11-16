"""Embudo del VAE sobre el dataset de funciones Python (ejercicio 2).

Cada corrida entrena un VAE con softmax agrupado y entropía cruzada categórica,
y al final (sin elegir checkpoints intermedios) mide:

- reconstrucción con z = mu en entrenamiento y validación;
- generación: 1.000 puntos fijos de N(0, I) decodificados, clasificados en
  compila / pertenece a la gramática / nueva textual / nueva semántica;
- KL final y cantidad de dimensiones latentes usadas.

Las corridas ya terminadas no se repiten, así cada paso del embudo agrega
corridas al mismo resumen.

Uso: uv run python -m sia_tp5.vae_code_study configs/vae_code_study.json [--only nombre ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .code_dataset import (SOURCE_LENGTH, belongs_to_grammar, compiles,
                           decode_patterns, function_parameters,
                           generate_function_sources, load_code_dataset,
                           semantic_signature)
from .optimizers import optimizer_from_config
from .training import fit_variational_autoencoder
from .variational_autoencoder import VariationalAutoencoder

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DIMENSION_THRESHOLD = 0.01


def build_model(run: dict, defaults: dict) -> VariationalAutoencoder:
    latent = run.get("latent", defaults["latent"])
    width = run.get("width", defaults["width"])
    return VariationalAutoencoder(
        [650, width, latent, width, 650], latent_layer=2,
        activations=["tanh", "linear", "tanh", "grouped_softmax"],
        loss="categorical_cross_entropy", categorical_groups=SOURCE_LENGTH,
        seed=run["seed"], init_scale=defaults["init_scale"],
        sampling_seed=run["seed"], kl_weight=run["beta"],
    )


def reconstruction(model, patterns, sources) -> dict:
    mu, _ = model.encode_distribution(patterns)
    decoded = decode_patterns(model.decode(mu))
    expected = np.asarray(sources)
    produced = np.asarray(decoded)
    characters = np.array([[a == b for a, b in zip(x, y)]
                           for x, y in zip(produced, expected)])
    return {
        "exact": float(np.mean(produced == expected)),
        "character_accuracy": float(characters.mean()),
        "grammar": float(np.mean([belongs_to_grammar(s) for s in decoded])),
    }


def kl_diagnostics(model, patterns) -> dict:
    mu, log_var = model.encode_distribution(patterns)
    per_dimension = np.mean(-0.5 * (1.0 + log_var - mu ** 2 - np.exp(log_var)), axis=0)
    return {
        "kl": float(per_dimension.sum()),
        "active_dimensions": int(np.sum(per_dimension > ACTIVE_DIMENSION_THRESHOLD)),
    }


def classify(source, known_sources, known_signatures) -> dict:
    grammar = belongs_to_grammar(source)
    novel_text = grammar and source not in known_sources
    novel_semantic = novel_text and semantic_signature(source) not in known_signatures
    return {"compiles": compiles(source), "grammar": grammar,
            "novel_text": novel_text, "novel_semantic": novel_semantic}


def prior_generation(model, dataset, *, count, seed) -> tuple[dict, list[dict]]:
    latent = model.architecture[model.latent_layer]
    z = np.random.default_rng(seed).standard_normal((count, latent))
    generated = decode_patterns(model.decode(z))
    known_sources = set(dataset.sources)
    known_signatures = {semantic_signature(s) for s in dataset.sources}
    rows = []
    for index, source in enumerate(generated):
        rows.append({"sample": index, "source": source,
                     **classify(source, known_sources, known_signatures)})
    valid = [row["source"] for row in rows if row["grammar"]]
    novel = [row["source"] for row in rows if row["novel_text"]]
    summary = {
        "prior_compile": float(np.mean([row["compiles"] for row in rows])),
        "prior_grammar": float(np.mean([row["grammar"] for row in rows])),
        "prior_novel_text": float(np.mean([row["novel_text"] for row in rows])),
        "prior_novel_semantic": float(np.mean([row["novel_semantic"] for row in rows])),
        "prior_unique_valid": len(set(valid)),
        "prior_unique_novel": len(set(novel)),
    }
    return summary, rows


def distribution_metrics(sources) -> dict:
    """Distancia de variación total entre las muestras válidas y la uniforme.

    El dataset se armó muestreando la familia de manera uniforme (y balanceada
    por par de operadores), así que un generador ideal produce cada par de
    operadores con probabilidad 1/9 y cada constante con probabilidad 1/100.
    """
    valid = [s for s in sources if belongs_to_grammar(s)]
    if not valid:
        return {"tv_operator_pairs": 1.0, "tv_constants": 1.0}
    parameters = [function_parameters(s) for s in valid]
    pairs = {}
    for left, _, right, _ in parameters:
        pairs[(left, right)] = pairs.get((left, right), 0) + 1
    pair_freq = np.array([pairs.get((a, b), 0) for a in "+-*" for b in "+-*"]) / len(valid)
    tv_pairs = 0.5 * np.abs(pair_freq - 1 / 9).sum()
    tv_constants = []
    for slot in (1, 3):
        counts = np.bincount([p[slot] for p in parameters], minlength=100) / len(valid)
        tv_constants.append(0.5 * np.abs(counts - 1 / 100).sum())
    return {"tv_operator_pairs": float(tv_pairs), "tv_constants": float(np.mean(tv_constants))}


def uniform_reference(dataset, *, count, repetitions=50, seed=7) -> dict:
    """Mismas métricas para `count` fuentes sorteadas uniformemente de la familia."""
    universe = generate_function_sources()
    known_sources = set(dataset.sources)
    known_signatures = {semantic_signature(s) for s in dataset.sources}
    generator = np.random.default_rng(seed)
    rows = []
    for _ in range(repetitions):
        sample = [universe[i] for i in generator.integers(0, len(universe), count)]
        classified = [classify(s, known_sources, known_signatures) for s in sample]
        rows.append({
            "prior_novel_text": np.mean([c["novel_text"] for c in classified]),
            "prior_novel_semantic": np.mean([c["novel_semantic"] for c in classified]),
            "prior_unique_valid": len(set(sample)),
            **distribution_metrics(sample),
        })
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_one(run: dict, defaults: dict, dataset, root: Path) -> dict:
    output = root / run["name"]
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(run, defaults)
    epochs = run.get("epochs", defaults["epochs"])
    start = perf_counter()
    history = fit_variational_autoencoder(
        model, dataset.train_patterns,
        optimizer=optimizer_from_config(
            {"name": "adam", "learning_rate": run.get("learning_rate",
                                                      defaults["learning_rate"])}),
        max_epochs=epochs,
        generator=np.random.default_rng(1000 + run["seed"]),
    )
    elapsed = perf_counter() - start
    train = reconstruction(model, dataset.train_patterns, dataset.train_sources)
    validation = reconstruction(model, dataset.validation_patterns,
                                dataset.validation_sources)
    prior, prior_rows = prior_generation(model, dataset, count=defaults["prior_samples"],
                                         seed=defaults["prior_seed"])
    summary = {
        "name": run["name"], "step": run.get("step", ""), "beta": run["beta"],
        "latent": model.architecture[model.latent_layer],
        "width": model.architecture[1], "seed": run["seed"], "epochs": epochs,
        "train_exact": train["exact"], "validation_exact": validation["exact"],
        "validation_character_accuracy": validation["character_accuracy"],
        "validation_grammar": validation["grammar"],
        **kl_diagnostics(model, dataset.train_patterns),
        "final_reconstruction_loss": history[-1]["reconstruction_loss"],
        **prior, "elapsed_seconds": elapsed,
    }
    if run.get("evaluate_test"):
        test = reconstruction(model, dataset.test_patterns, dataset.test_sources)
        summary.update({"test_exact": test["exact"],
                        "test_character_accuracy": test["character_accuracy"],
                        "test_grammar": test["grammar"]})
    model.save(output / "model.npz")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = ["epoch", "total_loss", "reconstruction_loss", "kl_loss"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in history)
    with (output / "prior-samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prior_rows[0].keys())
        writer.writeheader()
        writer.writerows(prior_rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--only", nargs="*", help="nombres de corridas a ejecutar")
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text())
    defaults = configuration["defaults"]
    root = ROOT / configuration["result_root"]
    dataset = load_code_dataset(split_seed=17)
    summaries = []
    for run in configuration["runs"]:
        if arguments.only and run["name"] not in arguments.only:
            continue
        summary = run_one(run, defaults, dataset, root)
        summaries.append(summary)
        print(f"{summary['name']}: val {summary['validation_exact']:.3f} "
              f"gram {summary['prior_grammar']:.3f} nueva {summary['prior_novel_text']:.3f} "
              f"KL {summary['kl']:.2f} dims {summary['active_dimensions']} "
              f"({summary['elapsed_seconds']:.0f} s)", flush=True)
    every = []
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        with (path.parent / "prior-samples.csv").open(encoding="utf-8") as handle:
            sources = [row["source"] for row in csv.DictReader(handle)]
        summary.update(distribution_metrics(sources))
        every.append(summary)
    reference = uniform_reference(dataset, count=defaults["prior_samples"])
    (ROOT / configuration["summary_output"]).with_suffix(".referencia.json").write_text(
        json.dumps(reference, indent=2) + "\n")
    print("referencia uniforme:", {k: round(v, 3) for k, v in reference.items()})
    output = ROOT / configuration["summary_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in every for key in row))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(every)
    print(f"resumen: {output}")


if __name__ == "__main__":
    main()

"""Comparar el AE control y pesos KL del VAE sobre funciones Python."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .code_dataset import (belongs_to_grammar, compiles, decode_patterns,
                           load_code_dataset)
from .optimizers import optimizer_from_config
from .training import (fit_autoencoder, fit_variational_autoencoder,
                       reconstruction_metrics)
from .variational_autoencoder import VariationalAutoencoder


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _source_metrics(model, patterns, expected_sources, *, variational):
    if variational:
        latent_size = model.architecture[model.latent_layer]
        epsilon = np.zeros((len(patterns), latent_size))
        probabilities = model.predict(patterns, epsilon)
    else:
        probabilities = model.predict(patterns)
    metrics = reconstruction_metrics(patterns, probabilities)
    decoded = decode_patterns(probabilities)
    return {
        "exact_sources": sum(
            actual == expected
            for actual, expected in zip(decoded, expected_sources)
        ),
        "compilable_sources": sum(compiles(source) for source in decoded),
        "grammar_sources": sum(
            belongs_to_grammar(source) for source in decoded
        ),
        "mean_bit_errors": metrics["mean_pixel_errors"],
        "max_bit_errors": metrics["max_pixel_errors"],
    }


def _prior_metrics(model, dataset, *, sample_count, seed):
    latent_size = model.architecture[model.latent_layer]
    latent = np.random.default_rng(seed).standard_normal(
        (sample_count, latent_size)
    )
    generated = decode_patterns(model.decode(latent))
    train_sources = set(dataset.train_sources)
    compilable = [compiles(source) for source in generated]
    grammar = [belongs_to_grammar(source) for source in generated]
    novel = [
        valid and source not in train_sources
        for source, valid in zip(generated, grammar)
    ]
    rows = [
        {
            "sample": index,
            "source": json.dumps(source),
            "compiles": compilable[index],
            "belongs_to_grammar": grammar[index],
            "novel_vs_train": novel[index],
        }
        for index, source in enumerate(generated)
    ]
    return {
        "prior_compile_rate": float(np.mean(compilable)),
        "prior_grammar_rate": float(np.mean(grammar)),
        "prior_novel_rate": float(np.mean(novel)),
        "prior_unique_sources": len(set(generated)),
    }, rows


def _plot_selected_history(path, history, kl_weight):
    epochs = [row["epoch"] for row in history]
    total = [row["total_loss"] for row in history]
    reconstruction = [row["reconstruction_loss"] for row in history]
    weighted_kl = [kl_weight * row["kl_loss"] for row in history]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(epochs, total, label="Total", alpha=0.85)
    axis.plot(epochs, reconstruction, label="Reconstrucción", alpha=0.85)
    axis.plot(epochs, weighted_kl, label="beta × KL", alpha=0.85)
    axis.set_xlabel("Época")
    axis.set_ylabel("Contribución a la pérdida")
    axis.set_yscale("log")
    axis.set_title(f"VAE de código seleccionado (beta = {kl_weight:g})")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(configuration):
    dataset = load_code_dataset(
        split_seed=configuration["split_seed"],
        train_fraction=configuration["train_fraction"],
    )
    result_root = Path(configuration["result_root"])
    if result_root.exists():
        raise FileExistsError(f"el estudio ya existe: {result_root}")
    result_root.mkdir(parents=True)
    summaries = []
    selected_history = None
    selected_model = None
    selected_prior_rows = None
    selected_kl_weight = None

    for run_config in configuration["runs"]:
        variational = run_config["type"] == "vae"
        common = dict(
            architecture=configuration["architecture"],
            latent_layer=configuration["latent_layer"],
            activations=configuration["activations"],
            loss="bce",
            seed=run_config["model_seed"],
            init_scale=configuration["init_scale"],
        )
        if variational:
            model = VariationalAutoencoder(
                **common,
                sampling_seed=run_config["training_seed"],
                kl_weight=run_config["kl_weight"],
            )
            history = fit_variational_autoencoder(
                model,
                dataset.train_patterns,
                optimizer=optimizer_from_config(configuration["optimizer"]),
                max_epochs=run_config["epochs"],
                generator=np.random.default_rng(run_config["training_seed"]),
            )
        else:
            model = Autoencoder(**common)
            history = fit_autoencoder(
                model,
                dataset.train_patterns,
                optimizer=optimizer_from_config(configuration["optimizer"]),
                max_epochs=run_config["epochs"],
                target_max_pixel_errors=0,
            )

        run_output = result_root / run_config["name"]
        run_output.mkdir()
        model.save(run_output / "model.npz")
        _write_csv(run_output / "history.csv", history)
        train_metrics = _source_metrics(
            model, dataset.train_patterns, dataset.train_sources,
            variational=variational,
        )
        test_metrics = _source_metrics(
            model, dataset.test_patterns, dataset.test_sources,
            variational=variational,
        )
        if variational:
            prior, prior_rows = _prior_metrics(
                model, dataset,
                sample_count=configuration["prior_sample_count"],
                seed=configuration["prior_seed"],
            )
            tail = history[-min(100, len(history)):]
            loss_metrics = {
                "tail_total_loss": float(np.mean([
                    row["total_loss"] for row in tail
                ])),
                "tail_reconstruction_loss": float(np.mean([
                    row["reconstruction_loss"] for row in tail
                ])),
                "tail_kl_loss": float(np.mean([
                    row["kl_loss"] for row in tail
                ])),
            }
            _write_csv(run_output / "prior-samples.csv", prior_rows)
        else:
            prior = {
                "prior_compile_rate": "",
                "prior_grammar_rate": "",
                "prior_novel_rate": "",
                "prior_unique_sources": "",
            }
            loss_metrics = {
                "tail_total_loss": "",
                "tail_reconstruction_loss": "",
                "tail_kl_loss": "",
            }

        summary = {
            "name": run_config["name"],
            "type": run_config["type"],
            "model_seed": run_config["model_seed"],
            "kl_weight": run_config.get("kl_weight", ""),
            "epochs_completed": history[-1]["epoch"],
            **loss_metrics,
            "train_exact_sources": train_metrics["exact_sources"],
            "train_mean_bit_errors": train_metrics["mean_bit_errors"],
            "test_exact_sources": test_metrics["exact_sources"],
            "test_mean_bit_errors": test_metrics["mean_bit_errors"],
            "test_compile_rate": test_metrics["compilable_sources"] / len(
                dataset.test_sources
            ),
            "test_grammar_rate": test_metrics["grammar_sources"] / len(
                dataset.test_sources
            ),
            **prior,
        }
        summaries.append(summary)
        (run_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if run_config["name"] == configuration["selected_run"]:
            selected_history = history
            selected_model = model
            selected_prior_rows = prior_rows
            selected_kl_weight = run_config["kl_weight"]

    if selected_model is None:
        raise ValueError("selected_run no coincide con ninguna corrida")
    _write_csv(configuration["comparison_output"], summaries)
    _write_csv(configuration["selected_prior_output"], selected_prior_rows)
    _plot_selected_history(
        configuration["selected_figure_output"], selected_history,
        selected_kl_weight,
    )
    selected_model.save(configuration["selected_model_output"])
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    summaries = run(configuration)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

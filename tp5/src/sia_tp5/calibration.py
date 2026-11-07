"""Calibrar tasas de optimizadores sin generar modelos ni figuras por corrida."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from .autoencoder import Autoencoder
from .data import load_font
from .optimizers import optimizer_from_config
from .training import fit_autoencoder, reconstruction_metrics


SUMMARY_FIELDS = (
    "run_id", "architecture", "parameters", "optimizer", "learning_rate",
    "seed", "status", "epochs_completed", "initial_loss", "final_loss",
    "loss_reduction", "minimum_loss", "minimum_loss_epoch",
    "final_minus_minimum_loss", "best_observed_max_pixel_errors",
    "best_observed_error_epoch", "first_requirement_epoch",
    "loss_at_half_budget", "loss_at_three_quarters_budget",
    "max_pixel_errors", "mean_pixel_errors",
    "exact_patterns", "patterns_with_at_most_one_error", "meets_requirement",
    "elapsed_seconds", "error",
)


def _rate_tag(rate):
    return format(float(rate), ".10g").replace(".", "p")


def _parameter_count(model):
    return int(sum(parameter.size for parameter in model.parameters))


def _write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _expanded_optimizer(base, learning_rate):
    configuration = dict(base)
    configuration["learning_rate"] = learning_rate
    return configuration


def planned_trials(configuration):
    """Expandir la configuracion sin entrenar ni escribir resultados."""
    trials = []
    for architecture_name in configuration["architectures"]:
        for optimizer in configuration["optimizers"]:
            for rate in optimizer["learning_rates"]:
                for seed in configuration["seeds"]:
                    trials.append({
                        "run_id": (
                            f"{architecture_name}__{optimizer['name']}"
                            f"__lr-{_rate_tag(rate)}__seed-{seed}"
                        ),
                        "architecture": architecture_name,
                        "optimizer": optimizer["name"],
                        "learning_rate": rate,
                        "seed": seed,
                        "max_epochs": configuration["max_epochs"],
                    })
    return trials


def _run_trial(dataset, architecture_name, architecture, optimizer_config,
               seed, max_epochs, requirement):
    rate = optimizer_config["learning_rate"]
    optimizer_name = optimizer_config["name"]
    run_id = (f"{architecture_name}__{optimizer_name}__lr-{_rate_tag(rate)}"
              f"__seed-{seed}")
    model = Autoencoder(
        architecture["layers"],
        latent_layer=architecture["latent_layer"],
        activations=architecture["activations"],
        seed=seed,
    )
    initial_loss = model.loss(dataset.patterns)
    start = perf_counter()
    history = []
    status = "completed"
    error = ""
    try:
        fit_autoencoder(
            model,
            dataset.patterns,
            optimizer=optimizer_from_config(optimizer_config),
            max_epochs=max_epochs,
            target_max_pixel_errors=None,
            progress=history.append,
        )
        predictions = model.predict(dataset.patterns)
        metrics = reconstruction_metrics(dataset.patterns, predictions)
        final_loss = model.loss(dataset.patterns)
    except (FloatingPointError, OverflowError) as exception:
        status = "diverged"
        error = str(exception)
        final_loss = None
        metrics = {
            "max_pixel_errors": None,
            "mean_pixel_errors": None,
            "exact_patterns": None,
            "patterns_with_at_most_one_error": None,
        }
    elapsed = perf_counter() - start
    minimum_loss_row = min(history, key=lambda row: row["loss"])
    best_error_row = min(
        history,
        key=lambda row: (
            row["max_pixel_errors"], row["mean_pixel_errors"], row["loss"],
        ),
    )
    first_requirement = next(
        (row for row in history
         if row["max_pixel_errors"] <= requirement),
        None,
    )
    by_epoch = {row["epoch"]: row for row in history}
    half_budget = by_epoch.get(max_epochs // 2)
    three_quarters_budget = by_epoch.get(3 * max_epochs // 4)
    summary = {
        "run_id": run_id,
        "architecture": architecture_name,
        "parameters": _parameter_count(model),
        "optimizer": optimizer_name,
        "learning_rate": rate,
        "seed": seed,
        "status": status,
        "epochs_completed": history[-1]["epoch"],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": (initial_loss - final_loss
                           if final_loss is not None else None),
        "minimum_loss": minimum_loss_row["loss"],
        "minimum_loss_epoch": minimum_loss_row["epoch"],
        "final_minus_minimum_loss": (
            final_loss - minimum_loss_row["loss"]
            if final_loss is not None else None
        ),
        "best_observed_max_pixel_errors": best_error_row["max_pixel_errors"],
        "best_observed_error_epoch": best_error_row["epoch"],
        "first_requirement_epoch": (
            first_requirement["epoch"] if first_requirement is not None else None
        ),
        "loss_at_half_budget": (
            half_budget["loss"] if half_budget is not None else None
        ),
        "loss_at_three_quarters_budget": (
            three_quarters_budget["loss"]
            if three_quarters_budget is not None else None
        ),
        "max_pixel_errors": metrics["max_pixel_errors"],
        "mean_pixel_errors": metrics["mean_pixel_errors"],
        "exact_patterns": metrics["exact_patterns"],
        "patterns_with_at_most_one_error": metrics[
            "patterns_with_at_most_one_error"
        ],
        "meets_requirement": (
            status == "completed"
            and metrics["max_pixel_errors"] <= requirement
        ),
        "elapsed_seconds": elapsed,
        "error": error,
    }
    history_rows = [{"run_id": run_id, **row} for row in history]
    return summary, history_rows


def run_calibration(configuration, output_root):
    """Ejecutar el producto pequeno definido en una configuracion de calibracion."""
    output = Path(output_root) / configuration["name"]
    if output.exists():
        raise FileExistsError(
            f"la calibracion ya existe y no se sobrescribira: {output}"
        )
    output.mkdir(parents=True)
    history_output = output / "histories"
    history_output.mkdir()
    (output / "spec.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    dataset = load_font()
    summaries = []
    for architecture_name, architecture in configuration["architectures"].items():
        for optimizer_base in configuration["optimizers"]:
            for rate in optimizer_base["learning_rates"]:
                optimizer_config = _expanded_optimizer(optimizer_base, rate)
                optimizer_config.pop("learning_rates")
                for seed in configuration["seeds"]:
                    summary, history = _run_trial(
                        dataset,
                        architecture_name,
                        architecture,
                        optimizer_config,
                        seed,
                        configuration["max_epochs"],
                        configuration["requirement_max_pixel_errors"],
                    )
                    summaries.append(summary)
                    if history:
                        _write_csv(
                            history_output / f"{summary['run_id']}.csv",
                            history,
                            history[0].keys(),
                        )

    _write_csv(output / "summary.csv", summaries, SUMMARY_FIELDS)
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true",
                        help="mostrar las corridas sin entrenar ni escribir archivos")
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.dry_run:
        trials = planned_trials(configuration)
        print(json.dumps({"runs": len(trials), "trials": trials}, indent=2))
        return
    output, summaries = run_calibration(configuration, arguments.output_root)
    completed = sum(row["status"] == "completed" for row in summaries)
    diverged = sum(row["status"] == "diverged" for row in summaries)
    print(json.dumps({
        "output": str(output),
        "runs": len(summaries),
        "completed": completed,
        "diverged": diverged,
    }, indent=2))


if __name__ == "__main__":
    main()

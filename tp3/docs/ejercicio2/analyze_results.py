"""Consolidar y graficar los experimentos del ejercicio 2."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEARNING_RATE_RUNS = (
    (0.001, "01-learning-rate/gd-lr-0001"),
    (0.01, "01-learning-rate/gd-lr-001"),
    (0.1, "01-learning-rate/gd-lr-01"),
    (0.3, "01b-learning-rate-upper/gd-lr-03"),
    (1.0, "01b-learning-rate-upper/gd-lr-1"),
    (3.0, "01c-learning-rate-limit/gd-lr-3"),
    (10.0, "01c-learning-rate-limit/gd-lr-10"),
)
ARCHITECTURE_RUNS = (
    (16, 12730, "02-architecture-width/arch-16"),
    (32, 25450, "01b-learning-rate-upper/gd-lr-1"),
    (64, 50890, "02-architecture-width/arch-64"),
)
BATCH_ETA_RUNS = tuple(
    (batch, learning_rate,
     f"03-batch-learning-rate/batch-{batch}-lr-{str(learning_rate).replace('.', '')}")
    for batch in (32, 64, 128)
    for learning_rate in (0.3, 1, 3)
)
LARGE_BATCH_ETA_RUNS = tuple(
    (batch, learning_rate,
     f"05-large-batch-learning-rate/batch-{batch}-lr-{str(learning_rate).replace('.', '')}")
    for batch in (256, 512, 1024)
    for learning_rate in (0.5, 1, 2)
)


def _read_history(folder):
    with (folder / "history.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    numeric = {}
    for name in rows[0]:
        if name in {"converged"}:
            continue
        numeric[name] = [float(row[name]) if row[name] else float("nan") for row in rows]
    return numeric


def _best(history):
    index = max(range(len(history["epoch"])),
                key=lambda item: history["validation_macro_f1_present"][item])
    return {
        "best_epoch": int(history["epoch"][index]),
        "training_mse": history["mse"][index],
        "validation_mse": history["validation_mse"][index],
        "training_accuracy": history["training_accuracy"][index],
        "validation_accuracy": history["validation_accuracy"][index],
        "training_macro_f1_present": history["training_macro_f1_present"][index],
        "validation_macro_f1_present": history["validation_macro_f1_present"][index],
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results_root, output):
    results_root, output = Path(results_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)

    rate_histories = []
    rate_rows = []
    for learning_rate, relative in LEARNING_RATE_RUNS:
        history = _read_history(results_root / relative)
        rate_histories.append((learning_rate, history))
        rate_rows.append({"learning_rate": learning_rate, **_best(history)})
    _write_csv(output / "learning-rates-summary.csv", rate_rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for learning_rate, history in rate_histories:
        label = f"η={learning_rate:g}"
        axes[0].plot(history["epoch"], history["validation_macro_f1_present"],
                     label=label)
        axes[1].plot(history["epoch"], history["validation_mse"], label=label)
    axes[0].set(title="Macro-F1 de validation", xlabel="Época", ylabel="Macro-F1")
    axes[1].set(title="MSE de validation", xlabel="Época", ylabel="MSE")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.suptitle("Ejercicio 2: comparación de learning rate")
    figure.tight_layout()
    figure.savefig(output / "learning-rates.png", dpi=180)
    plt.close(figure)

    architecture_histories = []
    architecture_rows = []
    for width, parameters, relative in ARCHITECTURE_RUNS:
        history = _read_history(results_root / relative)
        architecture_histories.append((width, history))
        architecture_rows.append({
            "hidden_width": width,
            "parameter_count": parameters,
            **_best(history),
        })
    _write_csv(output / "architecture-width-summary.csv", architecture_rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for width, history in architecture_histories:
        axes[0].plot(history["epoch"], history["training_macro_f1_present"],
                     linestyle="--", alpha=0.75, label=f"{width} train")
        axes[0].plot(history["epoch"], history["validation_macro_f1_present"],
                     label=f"{width} validation")
        axes[1].plot(history["epoch"], history["mse"],
                     linestyle="--", alpha=0.75, label=f"{width} train")
        axes[1].plot(history["epoch"], history["validation_mse"],
                     label=f"{width} validation")
    axes[0].set(title="Macro-F1 por ancho", xlabel="Época", ylabel="Macro-F1")
    axes[1].set(title="MSE por ancho", xlabel="Época", ylabel="MSE")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.suptitle("Ejercicio 2: capacidad y generalización")
    figure.tight_layout()
    figure.savefig(output / "architecture-width.png", dpi=180)
    plt.close(figure)

    batch_eta_rows = []
    for batch, learning_rate, relative in BATCH_ETA_RUNS:
        history = _read_history(results_root / relative)
        batch_eta_rows.append({
            "batch_size": batch,
            "learning_rate": learning_rate,
            **_best(history),
        })
    _write_csv(output / "batch-learning-rate-summary.csv", batch_eta_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for batch in (32, 64, 128):
        rows = [row for row in batch_eta_rows if row["batch_size"] == batch]
        axes[0].plot(
            [row["learning_rate"] for row in rows],
            [row["validation_macro_f1_present"] for row in rows],
            marker="o", label=f"batch {batch}")
        axes[1].plot(
            [row["learning_rate"] for row in rows],
            [row["best_epoch"] for row in rows],
            marker="o", label=f"batch {batch}")
    axes[0].set(title="Mejor macro-F1", xlabel="Learning rate", ylabel="Macro-F1")
    axes[1].set(title="Época del mejor valor", xlabel="Learning rate", ylabel="Época")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Interacción entre batch y learning rate (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "batch-learning-rate.png", dpi=180)
    plt.close(figure)

    groups = {}
    seed_histories = {}
    for relative_folder in (
            "04-batch-learning-rate-seeds", "04b-batch-eta-1-seeds"):
        seeds_folder = results_root / relative_folder
        with (seeds_folder / "summary.csv").open(newline="") as file:
            seed_runs = list(csv.DictReader(file))
        for row in seed_runs:
            base, seed = row["run"].rsplit("-seed-", 1)
            metrics = json.loads(
                (seeds_folder / row["run"] / "metrics.json").read_text())
            digit_five = next(item for item in metrics["validation"]["per_class"]
                              if item["label"] == 5)
            groups.setdefault(base, []).append({
                "seed": int(seed),
                "macro_f1": float(row["validation_macro_f1_present"]),
                "accuracy": float(row["validation_accuracy"]),
                "best_epoch": int(row["best_epoch"]),
                "seconds": float(row["seconds"]),
                "digit_5_f1": float(digit_five["f1"]),
            })
            seed_histories.setdefault(base, []).append(
                (int(seed), _read_history(seeds_folder / row["run"])))
    seed_rows = []
    for name, rows in groups.items():
        f1 = [row["macro_f1"] for row in rows]
        seed_rows.append({
            "configuration": name,
            "runs": len(rows),
            "macro_f1_mean": statistics.mean(f1),
            "macro_f1_stdev": statistics.stdev(f1),
            "macro_f1_min": min(f1),
            "macro_f1_max": max(f1),
            "accuracy_mean": statistics.mean(row["accuracy"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "best_epoch_median": statistics.median(row["best_epoch"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "batch-learning-rate-seeds-summary.csv", seed_rows)

    figure, axis = plt.subplots(figsize=(11, 5.2))
    names = [row["configuration"] for row in seed_rows]
    means = [row["macro_f1_mean"] for row in seed_rows]
    deviations = [row["macro_f1_stdev"] for row in seed_rows]
    axis.errorbar(names, means, yerr=deviations, fmt="o", capsize=6)
    for index, rows in enumerate(groups.values()):
        axis.scatter([index] * len(rows), [row["macro_f1"] for row in rows],
                     alpha=0.6, s=24)
    axis.set(title="Finalistas batch–eta en cinco semillas",
             ylabel="Mejor macro-F1 de validation", xlabel="Configuración")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output / "batch-learning-rate-seeds.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(seed_histories), 2, figsize=(14, 3.1 * len(seed_histories)),
        sharex=True, squeeze=False)
    colors = plt.cm.tab10.colors
    for row_index, (name, histories) in enumerate(seed_histories.items()):
        f1_axis, mse_axis = axes[row_index]
        for color, (seed, history) in zip(colors, sorted(histories)):
            epochs = history["epoch"][10:]
            f1_axis.plot(
                epochs, history["validation_macro_f1_present"][10:],
                color=color, alpha=0.55, linewidth=1, label=f"semilla {seed}")
            mse_axis.plot(
                epochs, history["validation_mse"][10:],
                color=color, alpha=0.55, linewidth=1, label=f"semilla {seed}")
        ordered = [history for _, history in sorted(histories)]
        epochs = ordered[0]["epoch"][10:]
        mean_f1 = [statistics.mean(history["validation_macro_f1_present"][index]
                                   for history in ordered)
                   for index in range(10, len(ordered[0]["epoch"]))]
        mean_mse = [statistics.mean(history["validation_mse"][index]
                                    for history in ordered)
                    for index in range(10, len(ordered[0]["epoch"]))]
        f1_axis.plot(epochs, mean_f1, color="black", linewidth=2.2, label="media")
        mse_axis.plot(epochs, mean_mse, color="black", linewidth=2.2, label="media")
        f1_axis.set_ylabel(f"{name}\nMacro-F1")
        mse_axis.set_ylabel("MSE")
        f1_axis.set_ylim(0.75, 0.95)
        mse_axis.set_yscale("log")
        for axis in (f1_axis, mse_axis):
            axis.grid(alpha=0.25)
        if row_index == 0:
            f1_axis.set_title("Macro-F1 de validation desde época 10")
            mse_axis.set_title("MSE de validation desde época 10")
            f1_axis.legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Época")
    axes[-1, 1].set_xlabel("Época")
    figure.suptitle("Ruido y variación entre semillas para batch–eta", y=1.002)
    figure.tight_layout()
    figure.savefig(output / "batch-learning-rate-seed-curves.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)

    large_summary = {
        row["run"]: row
        for row in json.loads(
            (results_root / "05-large-batch-learning-rate" / "summary.json").read_text())
    }
    large_rows = []
    large_histories = []
    for batch, learning_rate, relative in LARGE_BATCH_ETA_RUNS:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        large_rows.append({
            "batch_size": batch,
            "learning_rate": learning_rate,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"],
            "updates_at_best_epoch": large_summary[run_name]["updates_at_best_epoch"],
            "seconds": large_summary[run_name]["seconds"],
        })
        large_histories.append((batch, learning_rate, history))
    _write_csv(output / "large-batch-learning-rate-summary.csv", large_rows)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True, squeeze=False)
    colors = {0.5: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for row_index, batch in enumerate((256, 512, 1024)):
        updates_per_epoch = -(-9960 // batch)
        epoch_axis, update_axis = axes[row_index]
        for current_batch, learning_rate, history in large_histories:
            if current_batch != batch:
                continue
            epochs = history["epoch"][10:]
            macro_f1 = history["validation_macro_f1_present"][10:]
            label = f"η={learning_rate:g}"
            epoch_axis.plot(epochs, macro_f1, color=colors[learning_rate],
                            linewidth=1.1, alpha=0.85, label=label)
            update_axis.plot(
                [epoch * updates_per_epoch for epoch in epochs], macro_f1,
                color=colors[learning_rate], linewidth=1.1, alpha=0.85,
                label=label)
        epoch_axis.set_ylabel(f"batch {batch}\nMacro-F1")
        epoch_axis.set_xlabel("Época")
        update_axis.set_xlabel("Actualizaciones")
        for axis in (epoch_axis, update_axis):
            axis.set_ylim(0.78, 0.94)
            axis.grid(alpha=0.25)
        epoch_axis.legend(fontsize=8)
    axes[0, 0].set_title("Curvas según épocas")
    axes[0, 1].set_title("Las mismas curvas según actualizaciones")
    figure.suptitle("Batches grandes: interacción con learning rate", y=1.005)
    figure.tight_layout()
    figure.savefig(output / "large-batch-learning-rate-curves.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for batch in (256, 512, 1024):
        rows = [row for row in large_rows if row["batch_size"] == batch]
        learning_rates = [row["learning_rate"] for row in rows]
        axes[0].plot(learning_rates,
                     [row["validation_macro_f1_present"] for row in rows],
                     marker="o", label=f"batch {batch}")
        axes[1].plot(learning_rates, [row["digit_5_f1"] for row in rows],
                     marker="o", label=f"batch {batch}")
        axes[2].plot(learning_rates, [row["seconds"] for row in rows],
                     marker="o", label=f"batch {batch}")
    axes[0].set(title="Mejor macro-F1", ylabel="Macro-F1")
    axes[1].set(title="F1 del dígito 5", ylabel="F1")
    axes[2].set(title="Costo del horizonte completo", ylabel="Segundos")
    for axis in axes:
        axis.set_xlabel("Learning rate")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Resumen del barrido de batches grandes (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "large-batch-learning-rate-summary.png", dpi=180)
    plt.close(figure)

    for relative_folder in (
            "06-large-batch-finalists-seeds",
            "06b-mid-batch-lr-2-seeds"):
        seeds_folder = results_root / relative_folder
        with (seeds_folder / "summary.csv").open(newline="") as file:
            seed_runs = list(csv.DictReader(file))
        for row in seed_runs:
            base, seed = row["run"].rsplit("-seed-", 1)
            metrics = json.loads(
                (seeds_folder / row["run"] / "metrics.json").read_text())
            digit_five = next(item for item in metrics["validation"]["per_class"]
                              if item["label"] == 5)
            groups.setdefault(base, []).append({
                "seed": int(seed),
                "macro_f1": float(row["validation_macro_f1_present"]),
                "accuracy": float(row["validation_accuracy"]),
                "best_epoch": int(row["best_epoch"]),
                "seconds": float(row["seconds"]),
                "digit_5_f1": float(digit_five["f1"]),
            })
            seed_histories.setdefault(base, []).append(
                (int(seed), _read_history(seeds_folder / row["run"])))

    comparison_order = (
        "batch-32-lr-1", "batch-64-lr-1", "batch-64-lr-2",
        "batch-128-lr-1", "batch-128-lr-2", "batch-512-lr-05",
        "batch-512-lr-2", "batch-1024-lr-05",
    )
    comparison_rows = []
    for name in comparison_order:
        rows = groups[name]
        f1 = [row["macro_f1"] for row in rows]
        comparison_rows.append({
            "configuration": name,
            "runs": len(rows),
            "macro_f1_mean": statistics.mean(f1),
            "macro_f1_stdev": statistics.stdev(f1),
            "macro_f1_min": min(f1),
            "macro_f1_max": max(f1),
            "accuracy_mean": statistics.mean(row["accuracy"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "best_epoch_median": statistics.median(row["best_epoch"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "batch-eta-comprehensive-seeds-summary.csv", comparison_rows)

    figure, axis = plt.subplots(figsize=(13, 5.5))
    positions = list(range(len(comparison_rows)))
    axis.errorbar(
        positions,
        [row["macro_f1_mean"] for row in comparison_rows],
        yerr=[row["macro_f1_stdev"] for row in comparison_rows],
        fmt="o", color="black", capsize=6, label="media ± desvío")
    for position, name in zip(positions, comparison_order):
        axis.scatter(
            [position] * len(groups[name]),
            [row["macro_f1"] for row in groups[name]],
            alpha=0.55, s=28)
    axis.set_xticks(positions, comparison_order, rotation=18, ha="right")
    axis.set(title="Comparación consolidada de batch–eta en cinco semillas",
             ylabel="Mejor macro-F1 de validation", xlabel="Configuración")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "batch-eta-comprehensive-seeds.png", dpi=180)
    plt.close(figure)

    curve_order = (
        "batch-64-lr-1", "batch-64-lr-2",
        "batch-128-lr-1", "batch-128-lr-2",
    )
    figure, axes = plt.subplots(len(curve_order), 2, figsize=(14, 12),
                               sharex=True, squeeze=False)
    colors = plt.cm.tab10.colors
    for row_index, name in enumerate(curve_order):
        f1_axis, mse_axis = axes[row_index]
        histories = sorted(seed_histories[name])
        for color, (seed, history) in zip(colors, histories):
            epochs = history["epoch"][10:]
            f1_axis.plot(epochs, history["validation_macro_f1_present"][10:],
                         color=color, alpha=0.55, linewidth=1,
                         label=f"semilla {seed}")
            mse_axis.plot(epochs, history["validation_mse"][10:],
                          color=color, alpha=0.55, linewidth=1)
        ordered = [history for _, history in histories]
        epochs = ordered[0]["epoch"][10:]
        mean_f1 = [statistics.mean(history["validation_macro_f1_present"][index]
                                   for history in ordered)
                   for index in range(10, len(ordered[0]["epoch"]))]
        mean_mse = [statistics.mean(history["validation_mse"][index]
                                    for history in ordered)
                    for index in range(10, len(ordered[0]["epoch"]))]
        f1_axis.plot(epochs, mean_f1, color="black", linewidth=2.2, label="media")
        mse_axis.plot(epochs, mean_mse, color="black", linewidth=2.2)
        f1_axis.set_ylabel(f"{name}\nMacro-F1")
        mse_axis.set_ylabel("MSE")
        f1_axis.set_ylim(0.84, 0.95)
        mse_axis.set_yscale("log")
        for axis in (f1_axis, mse_axis):
            axis.grid(alpha=0.25)
        if row_index == 0:
            f1_axis.set_title("Macro-F1 de validation desde época 10")
            mse_axis.set_title("MSE de validation desde época 10")
            f1_axis.legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Época")
    axes[-1, 1].set_xlabel("Época")
    figure.suptitle("Refinación final de batch y eta", y=1.003)
    figure.tight_layout()
    figure.savefig(output / "batch-eta-finalist-curves.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "analysis-01-02")
    args = parser.parse_args()
    analyze(args.results_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

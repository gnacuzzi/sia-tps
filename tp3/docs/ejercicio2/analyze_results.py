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
FINAL_ARCHITECTURE_RUNS = (
    (16, 12730, "07-architecture-width-final/arch-16-batch-128-lr-2"),
    (32, 25450, "07-architecture-width-final/arch-32-batch-128-lr-2"),
    (64, 50890, "07-architecture-width-final/arch-64-batch-128-lr-2"),
    (128, 101770, "07b-architecture-width-upper/arch-128-batch-128-lr-2"),
    (256, 203530, "07b-architecture-width-upper/arch-256-batch-128-lr-2"),
)
DEPTH_RUNS = (
    ("[784,64,10]", "08-architecture-depth/arch-64-shallow"),
    ("[784,64,16,10]", "08-architecture-depth/arch-64-16-deep"),
    ("[784,64,32,10]", "08b-architecture-depth-wider/arch-64-32-deep"),
    ("[784,64,64,10]", "08c-architecture-depth-same-width/arch-64-64-deep"),
)
OPTIMIZER_RATE_RUNS = (
    ("GD", 0.5, "09-optimizer-learning-rate/gd-lr-05-final"),
    ("GD", 1.0, "09-optimizer-learning-rate/gd-lr-1-final"),
    ("GD", 2.0, "09-optimizer-learning-rate/gd-lr-2-final"),
    ("GD", 3.0, "09b-optimizer-learning-rate-boundaries/gd-lr-3-final"),
    ("Momentum", 0.01, "09b-optimizer-learning-rate-boundaries/momentum-lr-001"),
    ("Momentum", 0.02, "09b-optimizer-learning-rate-boundaries/momentum-lr-002"),
    ("Momentum", 0.05, "09-optimizer-learning-rate/momentum-lr-005"),
    ("Momentum", 0.1, "09-optimizer-learning-rate/momentum-lr-01"),
    ("Momentum", 0.2, "09-optimizer-learning-rate/momentum-lr-02"),
    ("Adaptativo", 0.5, "09-optimizer-learning-rate/adaptive-lr-05"),
    ("Adaptativo", 1.0, "09-optimizer-learning-rate/adaptive-lr-1"),
    ("Adaptativo", 2.0, "09-optimizer-learning-rate/adaptive-lr-2"),
    ("RMSProp", 0.0001, "09-optimizer-learning-rate/rmsprop-lr-00001"),
    ("RMSProp", 0.001, "09-optimizer-learning-rate/rmsprop-lr-0001"),
    ("RMSProp", 0.01, "09-optimizer-learning-rate/rmsprop-lr-001"),
    ("RMSProp", 0.03, "09b-optimizer-learning-rate-boundaries/rmsprop-lr-003"),
    ("RMSProp", 0.1, "09b-optimizer-learning-rate-boundaries/rmsprop-lr-01"),
    ("Adam", 0.0001, "09-optimizer-learning-rate/adam-lr-00001"),
    ("Adam", 0.001, "09-optimizer-learning-rate/adam-lr-0001"),
    ("Adam", 0.01, "09-optimizer-learning-rate/adam-lr-001"),
    ("Adam", 0.03, "09b-optimizer-learning-rate-boundaries/adam-lr-003"),
    ("Adam", 0.1, "09b-optimizer-learning-rate-boundaries/adam-lr-01"),
)
OPTIMIZER_FINALISTS = (
    ("GD η=2", "gd-lr-2"),
    ("GD η=3", "gd-lr-3"),
    ("Adaptativo η₀=1", "adaptive-lr-1"),
    ("RMSProp η=0.01", "rmsprop-lr-001"),
    ("Adam η=0.001", "adam-lr-0001"),
    ("Adam η=0.01", "adam-lr-001"),
)
ARCHITECTURE_OPTIMIZER_RUNS = tuple(
    (architecture, optimizer,
     f"11-architecture-optimizer-interaction/arch-{architecture}-{optimizer.lower()}")
    for architecture in ("32", "64", "128", "256", "64-64")
    for optimizer in ("Adaptive", "RMSProp", "Adam")
)
ARCHITECTURE_OPTIMIZER_LOCAL_RUNS = (
    ("128", "Adaptive", 0.5,
     "11b-architecture-optimizer-local-rates/arch-128-adaptive-lr-05"),
    ("128", "Adam", 0.001,
     "11b-architecture-optimizer-local-rates/arch-128-adam-lr-0001"),
    ("128", "Adam", 0.003,
     "11b-architecture-optimizer-local-rates/arch-128-adam-lr-0003"),
    ("256", "Adaptive", 0.25,
     "11b-architecture-optimizer-local-rates/arch-256-adaptive-lr-025"),
    ("256", "Adaptive", 0.5,
     "11b-architecture-optimizer-local-rates/arch-256-adaptive-lr-05"),
    ("256", "RMSProp", 0.001,
     "11b-architecture-optimizer-local-rates/arch-256-rmsprop-lr-0001"),
    ("256", "RMSProp", 0.003,
     "11b-architecture-optimizer-local-rates/arch-256-rmsprop-lr-0003"),
    ("256", "Adam", 0.001,
     "11b-architecture-optimizer-local-rates/arch-256-adam-lr-0001"),
    ("256", "Adam", 0.003,
     "11b-architecture-optimizer-local-rates/arch-256-adam-lr-0003"),
    ("64-64", "RMSProp", 0.001,
     "11b-architecture-optimizer-local-rates/arch-64-64-rmsprop-lr-0001"),
    ("64-64", "RMSProp", 0.003,
     "11b-architecture-optimizer-local-rates/arch-64-64-rmsprop-lr-0003"),
)
ADAM_BATCH_RATE_RUNS = tuple(
    (batch, learning_rate,
     f"13-adam-batch-learning-rate/batch-{batch}-lr-{tag}")
    for batch in (32, 64, 128, 256, 512)
    for learning_rate, tag in ((0.001, "0001"), (0.003, "0003"),
                               (0.01, "001"))
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


def _at_epoch(history, epoch):
    """Recuperar la fila elegida por el runner, incluido su desempate por MSE."""
    index = history["epoch"].index(epoch)
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

    architecture_summary = {}
    for relative_folder in (
            "07-architecture-width-final", "07b-architecture-width-upper"):
        for row in json.loads((results_root / relative_folder / "summary.json").read_text()):
            architecture_summary[row["run"]] = row
    final_architecture_rows = []
    final_architecture_histories = []
    for width, parameters, relative in FINAL_ARCHITECTURE_RUNS:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        final_architecture_rows.append({
            "hidden_width": width,
            "parameter_count": parameters,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"],
            "seconds": architecture_summary[run_name]["seconds"],
        })
        final_architecture_histories.append((width, history))
    _write_csv(output / "architecture-width-final-summary.csv",
               final_architecture_rows)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {
        16: "tab:blue", 32: "tab:orange", 64: "tab:green",
        128: "tab:red", 256: "tab:purple",
    }
    for width, history in final_architecture_histories:
        color = colors[width]
        axes[0, 0].plot(history["epoch"], history["training_macro_f1_present"],
                        color=color, linestyle="--", alpha=0.65,
                        label=f"{width} train")
        axes[0, 0].plot(history["epoch"], history["validation_macro_f1_present"],
                        color=color, label=f"{width} validation")
        axes[0, 1].plot(history["epoch"], history["mse"], color=color,
                        linestyle="--", alpha=0.65, label=f"{width} train")
        axes[0, 1].plot(history["epoch"], history["validation_mse"],
                        color=color, label=f"{width} validation")
        if width != 256:
            axes[1, 0].plot(history["epoch"][10:],
                            history["validation_macro_f1_present"][10:],
                            color=color, label=f"{width} neuronas")
    widths = [row["hidden_width"] for row in final_architecture_rows]
    axes[1, 1].plot(
        widths,
        [row["validation_macro_f1_present"] for row in final_architecture_rows],
        marker="o", label="Macro-F1")
    axes[1, 1].plot(widths, [row["digit_5_f1"] for row in final_architecture_rows],
                    marker="o", label="F1 del 5")
    axes[0, 0].set(title="Macro-F1 de training y validation", xlabel="Época",
                   ylabel="Macro-F1")
    axes[0, 1].set(title="MSE de training y validation", xlabel="Época",
                   ylabel="MSE")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set(title="Anchos competitivos: validation desde época 10",
                   xlabel="Época", ylabel="Macro-F1", ylim=(0.82, 0.94))
    axes[1, 1].set(title="Mejor resultado por ancho", xlabel="Neuronas ocultas",
                   ylabel="F1", xticks=widths)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Arquitectura con batch=128 y η=2 (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "architecture-width-final.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    all_widths = [row["hidden_width"] for row in final_architecture_rows]
    all_macro_f1 = [row["validation_macro_f1_present"]
                    for row in final_architecture_rows]
    all_digit_five = [row["digit_5_f1"] for row in final_architecture_rows]
    axes[0].plot(all_widths, all_macro_f1, marker="o", label="Macro-F1")
    axes[0].plot(all_widths, all_digit_five, marker="o", label="F1 del 5")
    competitive_widths = all_widths[:-1]
    axes[1].plot(competitive_widths, all_macro_f1[:-1], marker="o",
                 label="Macro-F1")
    axes[1].plot(competitive_widths, all_digit_five[:-1], marker="o",
                 label="F1 del 5")
    for axis in axes:
        axis.set_xlabel("Neuronas ocultas")
        axis.set_ylabel("F1")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set(title="Todos los anchos", xticks=all_widths, ylim=(-0.02, 1.0))
    axes[1].set(title="Zoom de los anchos competitivos",
                xlim=(12, 132), xticks=competitive_widths, ylim=(0.70, 0.95))
    figure.suptitle("Mejor resultado de validation por ancho (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "architecture-width-best.png", dpi=180)
    plt.close(figure)

    depth_summary = {}
    for relative_folder in (
            "08-architecture-depth", "08b-architecture-depth-wider",
            "08c-architecture-depth-same-width"):
        for row in json.loads((results_root / relative_folder / "summary.json").read_text()):
            depth_summary[row["run"]] = row
    depth_rows = []
    depth_histories = []
    depth_per_class = []
    for architecture, relative in DEPTH_RUNS:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        depth_rows.append({
            "architecture": architecture,
            "parameter_count": depth_summary[run_name]["parameter_count"],
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"],
            "seconds": depth_summary[run_name]["seconds"],
        })
        depth_histories.append((architecture, history))
        depth_per_class.append((architecture, {
            item["label"]: item["f1"]
            for item in metrics["validation"]["per_class"]
            if item["support"] > 0
        }))
    _write_csv(output / "architecture-depth-summary.csv", depth_rows)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ("tab:green", "tab:purple", "tab:orange", "tab:red")
    for color, (architecture, history) in zip(colors, depth_histories):
        axes[0, 0].plot(history["epoch"], history["training_macro_f1_present"],
                        color=color, linestyle="--", alpha=0.7,
                        label=f"{architecture} train")
        axes[0, 0].plot(history["epoch"], history["validation_macro_f1_present"],
                        color=color, label=f"{architecture} validation")
        axes[0, 1].plot(history["epoch"], history["mse"], color=color,
                        linestyle="--", alpha=0.7,
                        label=f"{architecture} train")
        axes[0, 1].plot(history["epoch"], history["validation_mse"],
                        color=color, label=f"{architecture} validation")
        axes[1, 0].plot(history["epoch"][10:],
                        history["validation_macro_f1_present"][10:],
                        color=color, label=architecture)
    labels = sorted(depth_per_class[0][1])
    positions = list(range(len(labels)))
    bar_width = 0.20
    for offset, (color, (architecture, values)) in enumerate(
            zip(colors, depth_per_class)):
        axes[1, 1].bar(
            [position + (offset - (len(depth_per_class) - 1) / 2) * bar_width
             for position in positions],
            [values[label] for label in labels], width=bar_width, color=color,
            alpha=0.8, label=architecture)
    axes[0, 0].set(title="Macro-F1 de training y validation", xlabel="Época",
                   ylabel="Macro-F1")
    axes[0, 1].set(title="MSE de training y validation", xlabel="Época",
                   ylabel="MSE")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set(title="Validation desde época 10", xlabel="Época",
                   ylabel="Macro-F1", ylim=(0.82, 0.95))
    axes[1, 1].set(title="F1 de validation por dígito", xlabel="Dígito",
                   ylabel="F1", ylim=(0.65, 1.0))
    axes[1, 1].set_xticks(positions, labels)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Efecto de agregar una segunda capa oculta (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "architecture-depth.png", dpi=180)
    plt.close(figure)

    optimizer_rows = []
    optimizer_histories = []
    optimizer_summaries = {}
    for folder in ("09-optimizer-learning-rate",
                   "09b-optimizer-learning-rate-boundaries"):
        for row in json.loads((results_root / folder / "summary.json").read_text()):
            optimizer_summaries[row["run"]] = row
    for optimizer, learning_rate, relative in OPTIMIZER_RATE_RUNS:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        optimizer_rows.append({
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"] or 0.0,
            "digit_5_recall": digit_five["recall"] or 0.0,
            "ignored_digit_5": digit_five["true_positive"] == 0,
            "seconds": optimizer_summaries[run_name]["seconds"],
        })
        optimizer_histories.append((optimizer, learning_rate, history))
    _write_csv(output / "optimizer-learning-rate-summary.csv", optimizer_rows)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flat
    for axis, optimizer in zip(axes, ("GD", "Momentum", "Adaptativo",
                                      "RMSProp", "Adam")):
        for candidate, learning_rate, history in optimizer_histories:
            if candidate == optimizer:
                axis.plot(history["epoch"],
                          history["validation_macro_f1_present"],
                          label=f"η={learning_rate:g}")
        axis.set(title=optimizer, xlabel="Época", ylabel="Macro-F1 validation",
                 ylim=(-0.02, 1.0))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[5].axis("off")
    figure.suptitle("Búsqueda de learning rate por optimizador (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "optimizer-learning-rate-curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for optimizer in ("GD", "Momentum", "Adaptativo", "RMSProp", "Adam"):
        rows = [row for row in optimizer_rows if row["optimizer"] == optimizer]
        rows.sort(key=lambda row: row["learning_rate"])
        axes[0].plot([row["learning_rate"] for row in rows],
                     [row["validation_macro_f1_present"] for row in rows],
                     marker="o", label=optimizer)
        axes[1].plot([row["learning_rate"] for row in rows],
                     [row["digit_5_f1"] for row in rows], marker="o",
                     label=optimizer)
    for axis, title, ylabel in zip(
            axes, ("Macro-F1", "F1 del dígito 5"), ("Macro-F1", "F1")):
        axis.set_xscale("log")
        axis.set(title=title, xlabel="Learning rate (escala log)", ylabel=ylabel,
                 ylim=(-0.02, 1.0))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Resultado en la mejor época de cada corrida (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "optimizer-learning-rate-selection.png", dpi=180)
    plt.close(figure)

    finalists_root = results_root / "10-optimizer-finalists-seeds"
    finalist_summaries = {
        row["run"]: row
        for row in json.loads((finalists_root / "summary.json").read_text())
    }
    finalist_seed_rows = []
    finalist_histories = {}
    for label, base_name in OPTIMIZER_FINALISTS:
        finalist_histories[label] = []
        for seed in range(5):
            run_name = f"{base_name}-seed-{seed}"
            run_folder = finalists_root / run_name
            history = _read_history(run_folder)
            finalist_histories[label].append(history)
            best = _best(history)
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            finalist_seed_rows.append({
                "candidate": label,
                "seed": seed,
                **best,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "digit_5_f1": digit_five["f1"] or 0.0,
                "digit_5_recall": digit_five["recall"] or 0.0,
                "ignored_digit_5": digit_five["true_positive"] == 0,
                "seconds": finalist_summaries[run_name]["seconds"],
            })
    _write_csv(output / "optimizer-finalists-seeds.csv", finalist_seed_rows)

    aggregate_rows = []
    for label, _ in OPTIMIZER_FINALISTS:
        rows = [row for row in finalist_seed_rows if row["candidate"] == label]
        aggregate_rows.append({
            "candidate": label,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "ignored_digit_5_runs": sum(row["ignored_digit_5"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "optimizer-finalists-summary.csv", aggregate_rows)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    labels = [label for label, _ in OPTIMIZER_FINALISTS]
    positions = list(range(len(labels)))
    for position, label in enumerate(labels):
        rows = [row for row in finalist_seed_rows if row["candidate"] == label]
        axes[0, 0].scatter([position] * 5,
                           [row["validation_macro_f1_present"] for row in rows])
        axes[0, 1].scatter([position] * 5,
                           [row["digit_5_f1"] for row in rows])
    axes[0, 0].boxplot([
        [row["validation_macro_f1_present"] for row in finalist_seed_rows
         if row["candidate"] == label] for label in labels], positions=positions)
    axes[0, 1].boxplot([
        [row["digit_5_f1"] for row in finalist_seed_rows
         if row["candidate"] == label] for label in labels], positions=positions)
    axes[1, 0].bar(positions, [row["gap_mean"] for row in aggregate_rows])
    axes[1, 1].bar(positions, [row["best_epoch_mean"] for row in aggregate_rows])
    axes[0, 0].set(title="Macro-F1 en cinco semillas", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5 en cinco semillas", ylabel="F1")
    axes[1, 0].set(title="Gap medio train − validation", ylabel="Gap")
    axes[1, 1].set(title="Mejor época media", ylabel="Época")
    for axis in axes.flat:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Estabilidad de los optimizadores finalistas")
    figure.tight_layout()
    figure.savefig(output / "optimizer-finalists-stability.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for axis, label in zip(axes.flat, labels):
        histories = finalist_histories[label]
        epochs = histories[0]["epoch"]
        for metric, linestyle, color in (
                ("training_macro_f1_present", "--", "tab:blue"),
                ("validation_macro_f1_present", "-", "tab:orange")):
            means = [statistics.mean(history[metric][index]
                                     for history in histories)
                     for index in range(len(epochs))]
            deviations = [statistics.stdev(history[metric][index]
                                            for history in histories)
                          for index in range(len(epochs))]
            axis.plot(epochs, means, color=color, linestyle=linestyle,
                      label="train" if metric.startswith("training") else "validation")
            axis.fill_between(epochs,
                              [mean - deviation for mean, deviation
                               in zip(means, deviations)],
                              [mean + deviation for mean, deviation
                               in zip(means, deviations)],
                              color=color, alpha=0.15)
        axis.set(title=label, xlabel="Época", ylabel="Macro-F1", ylim=(0.65, 1.01))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Curvas medias ± un desvío estándar (cinco semillas)")
    figure.tight_layout()
    figure.savefig(output / "optimizer-finalists-curves.png", dpi=180)
    plt.close(figure)

    interaction_summary = {
        row["run"]: row for row in json.loads((
            results_root / "11-architecture-optimizer-interaction" / "summary.json"
        ).read_text())
    }
    interaction_rows = []
    for architecture, optimizer, relative in ARCHITECTURE_OPTIMIZER_RUNS:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        interaction_rows.append({
            "architecture": architecture,
            "optimizer": optimizer,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"] or 0.0,
            "ignored_digit_5": digit_five["true_positive"] == 0,
            "seconds": interaction_summary[run_name]["seconds"],
        })
    _write_csv(output / "architecture-optimizer-interaction.csv",
               interaction_rows)

    architecture_labels = ["32", "64", "128", "256", "64-64"]
    positions = list(range(len(architecture_labels)))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for optimizer in ("Adaptive", "RMSProp", "Adam"):
        rows = [next(row for row in interaction_rows
                     if row["architecture"] == architecture
                     and row["optimizer"] == optimizer)
                for architecture in architecture_labels]
        axes[0, 0].plot(positions,
                        [row["validation_macro_f1_present"] for row in rows],
                        marker="o", label=optimizer)
        axes[0, 1].plot(positions, [row["digit_5_f1"] for row in rows],
                        marker="o", label=optimizer)
        axes[1, 0].plot(positions, [row["best_epoch"] for row in rows],
                        marker="o", label=optimizer)
        axes[1, 1].plot(positions, [row["seconds"] for row in rows],
                        marker="o", label=optimizer)
    axes[0, 0].set(title="Macro-F1 en la mejor época", ylabel="Macro-F1",
                   ylim=(-0.02, 1.0))
    axes[0, 1].set(title="F1 del dígito 5", ylabel="F1", ylim=(-0.02, 1.0))
    axes[1, 0].set(title="Mejor época", ylabel="Época")
    axes[1, 1].set(title="Tiempo total", ylabel="Segundos")
    for axis in axes.flat:
        axis.set_xticks(positions, architecture_labels)
        axis.set_xlabel("Arquitectura oculta")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Interacción arquitectura–optimizador (tasas transferidas, semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "architecture-optimizer-interaction.png", dpi=180)
    plt.close(figure)

    local_summary = {
        row["run"]: row for row in json.loads((
            results_root / "11b-architecture-optimizer-local-rates" / "summary.json"
        ).read_text())
    }
    local_rows = []
    for architecture, optimizer, learning_rate, relative in (
            ARCHITECTURE_OPTIMIZER_LOCAL_RUNS):
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        local_rows.append({
            "architecture": architecture,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"] or 0.0,
            "ignored_digit_5": digit_five["true_positive"] == 0,
            "seconds": local_summary[run_name]["seconds"],
        })
    _write_csv(output / "architecture-optimizer-local-rates.csv", local_rows)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    local_labels = [
        f"{row['architecture']}\n{row['optimizer']} η={row['learning_rate']:g}"
        for row in local_rows]
    local_positions = list(range(len(local_rows)))
    axes[0].bar(local_positions,
                [row["validation_macro_f1_present"] for row in local_rows])
    axes[1].bar(local_positions, [row["digit_5_f1"] for row in local_rows])
    axes[0].set(title="Macro-F1", ylabel="Macro-F1", ylim=(0, 1.0))
    axes[1].set(title="F1 del dígito 5", ylabel="F1", ylim=(0, 1.0))
    for axis in axes:
        axis.set_xticks(local_positions, local_labels, rotation=45, ha="right")
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Ajuste local de tasas en combinaciones problemáticas (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "architecture-optimizer-local-rates.png", dpi=180)
    plt.close(figure)

    combined_finalists = (
        ("64 + RMSProp η=0.01", "10-optimizer-finalists-seeds",
         "rmsprop-lr-001"),
        ("128 + RMSProp η=0.01", "12-architecture-optimizer-finalists-seeds",
         "arch-128-rmsprop-lr-001"),
        ("128 + Adam η=0.003", "12-architecture-optimizer-finalists-seeds",
         "arch-128-adam-lr-0003"),
        ("64-64 + Adam η=0.01", "12-architecture-optimizer-finalists-seeds",
         "arch-64-64-adam-lr-001"),
    )
    architecture_finalist_rows = []
    architecture_finalist_histories = {}
    for label, folder, base_name in combined_finalists:
        folder_path = results_root / folder
        summaries = {
            row["run"]: row
            for row in json.loads((folder_path / "summary.json").read_text())
        }
        architecture_finalist_histories[label] = []
        for seed in range(5):
            run_name = f"{base_name}-seed-{seed}"
            run_folder = folder_path / run_name
            history = _read_history(run_folder)
            architecture_finalist_histories[label].append(history)
            best = _best(history)
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            architecture_finalist_rows.append({
                "candidate": label,
                "seed": seed,
                **best,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "digit_5_f1": digit_five["f1"] or 0.0,
                "ignored_digit_5": digit_five["true_positive"] == 0,
                "seconds": summaries[run_name]["seconds"],
            })
    _write_csv(output / "architecture-optimizer-finalists-seeds.csv",
               architecture_finalist_rows)

    architecture_aggregate_rows = []
    final_labels = [item[0] for item in combined_finalists]
    for label in final_labels:
        rows = [row for row in architecture_finalist_rows
                if row["candidate"] == label]
        architecture_aggregate_rows.append({
            "candidate": label,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "ignored_digit_5_runs": sum(row["ignored_digit_5"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "architecture-optimizer-finalists-summary.csv",
               architecture_aggregate_rows)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    final_positions = list(range(len(final_labels)))
    for position, label in enumerate(final_labels):
        rows = [row for row in architecture_finalist_rows
                if row["candidate"] == label]
        axes[0, 0].scatter([position] * 5,
                           [row["validation_macro_f1_present"] for row in rows])
        axes[0, 1].scatter([position] * 5,
                           [row["digit_5_f1"] for row in rows])
    axes[0, 0].boxplot([
        [row["validation_macro_f1_present"] for row in architecture_finalist_rows
         if row["candidate"] == label] for label in final_labels],
        positions=final_positions)
    axes[0, 1].boxplot([
        [row["digit_5_f1"] for row in architecture_finalist_rows
         if row["candidate"] == label] for label in final_labels],
        positions=final_positions)
    axes[1, 0].bar(final_positions,
                   [row["gap_mean"] for row in architecture_aggregate_rows])
    axes[1, 1].bar(final_positions,
                   [row["seconds_mean"] for row in architecture_aggregate_rows])
    axes[0, 0].set(title="Macro-F1 en cinco semillas", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5", ylabel="F1")
    axes[1, 0].set(title="Gap medio train − validation", ylabel="Gap")
    axes[1, 1].set(title="Tiempo medio de 200 épocas", ylabel="Segundos")
    for axis in axes.flat:
        axis.set_xticks(final_positions, final_labels, rotation=20, ha="right")
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Finalistas arquitectura–optimizador")
    figure.tight_layout()
    figure.savefig(output / "architecture-optimizer-finalists.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, label in zip(axes.flat, final_labels):
        histories = architecture_finalist_histories[label]
        epochs = histories[0]["epoch"]
        for metric, linestyle, color in (
                ("training_macro_f1_present", "--", "tab:blue"),
                ("validation_macro_f1_present", "-", "tab:orange")):
            means = [statistics.mean(history[metric][index]
                                     for history in histories)
                     for index in range(len(epochs))]
            deviations = [statistics.stdev(history[metric][index]
                                            for history in histories)
                          for index in range(len(epochs))]
            axis.plot(epochs, means, color=color, linestyle=linestyle,
                      label="train" if metric.startswith("training") else "validation")
            axis.fill_between(
                epochs,
                [mean - deviation for mean, deviation in zip(means, deviations)],
                [mean + deviation for mean, deviation in zip(means, deviations)],
                color=color, alpha=0.15)
        axis.set(title=label, xlabel="Época", ylabel="Macro-F1", ylim=(0.65, 1.01))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Curvas medias ± un desvío: arquitectura–optimizador")
    figure.tight_layout()
    figure.savefig(output / "architecture-optimizer-finalist-curves.png", dpi=180)
    plt.close(figure)

    adam_batch_summary = {
        row["run"]: row for row in json.loads((
            results_root / "13-adam-batch-learning-rate" / "summary.json"
        ).read_text())
    }
    adam_batch_rows = []
    adam_batch_histories = []
    for batch, learning_rate, relative in ADAM_BATCH_RATE_RUNS:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        adam_batch_rows.append({
            "batch": batch,
            "learning_rate": learning_rate,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"] or 0.0,
            "ignored_digit_5": digit_five["true_positive"] == 0,
            "updates_at_best_epoch": adam_batch_summary[run_name][
                "updates_at_best_epoch"],
            "seconds": adam_batch_summary[run_name]["seconds"],
        })
        adam_batch_histories.append((batch, learning_rate, history))
    _write_csv(output / "adam-batch-learning-rate-summary.csv",
               adam_batch_rows)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for axis, batch in zip(axes.flat, (32, 64, 128, 256, 512)):
        for candidate_batch, learning_rate, history in adam_batch_histories:
            if candidate_batch == batch:
                axis.plot(history["epoch"],
                          history["validation_macro_f1_present"],
                          label=f"η={learning_rate:g}")
        axis.set(title=f"Batch {batch}", xlabel="Época",
                 ylabel="Macro-F1 validation", ylim=(-0.02, 1.0))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes.flat[5].axis("off")
    figure.suptitle("Interacción batch–learning rate con Adam (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "adam-batch-learning-rate-curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for learning_rate in (0.001, 0.003, 0.01):
        rows = [row for row in adam_batch_rows
                if row["learning_rate"] == learning_rate]
        rows.sort(key=lambda row: row["batch"])
        axes[0].plot([row["batch"] for row in rows],
                     [row["validation_macro_f1_present"] for row in rows],
                     marker="o", label=f"η={learning_rate:g}")
        axes[1].plot([row["batch"] for row in rows],
                     [row["digit_5_f1"] for row in rows], marker="o",
                     label=f"η={learning_rate:g}")
    axes[0].set(title="Macro-F1 en la mejor época", ylabel="Macro-F1")
    axes[1].set(title="F1 del dígito 5", ylabel="F1")
    for axis in axes:
        axis.set(xlabel="Batch", ylim=(-0.02, 1.0), xticks=(32, 64, 128, 256, 512))
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Mejor resultado por batch y learning rate (semilla 0)")
    figure.tight_layout()
    figure.savefig(output / "adam-batch-learning-rate-summary.png", dpi=180)
    plt.close(figure)

    extension_rows = []
    extension_histories = []
    extension_root = results_root / "13b-adam-batch-512-extension"
    extension_summary = {
        row["run"]: row
        for row in json.loads((extension_root / "summary.json").read_text())
    }
    for learning_rate, run_name in (
            (0.003, "batch-512-lr-0003-extended"),
            (0.01, "batch-512-lr-001-extended")):
        run_folder = extension_root / run_name
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        extension_rows.append({
            "batch": 512,
            "learning_rate": learning_rate,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"] or 0.0,
            "updates_at_best_epoch": extension_summary[run_name][
                "updates_at_best_epoch"],
            "seconds": extension_summary[run_name]["seconds"],
        })
        extension_histories.append((learning_rate, history))
    _write_csv(output / "adam-batch-512-extension-summary.csv", extension_rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for learning_rate, history in extension_histories:
        axes[0].plot(history["epoch"],
                     history["validation_macro_f1_present"],
                     label=f"η={learning_rate:g}")
        axes[1].plot(history["epoch"], history["validation_mse"],
                     label=f"η={learning_rate:g}")
    axes[0].set(title="Macro-F1 de validation", xlabel="Época", ylabel="Macro-F1")
    axes[1].set(title="MSE de validation", xlabel="Época", ylabel="MSE")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Extensión de batch 512 hasta 600 épocas")
    figure.tight_layout()
    figure.savefig(output / "adam-batch-512-extension.png", dpi=180)
    plt.close(figure)

    batch_finalists = (
        ("Batch 64, η=0.003", "14-adam-batch-finalists-seeds",
         "batch-64-lr-0003"),
        ("Batch 128, η=0.003", "12-architecture-optimizer-finalists-seeds",
         "arch-128-adam-lr-0003"),
        ("Batch 512, η=0.01", "14-adam-batch-finalists-seeds",
         "batch-512-lr-001"),
    )
    batch_finalist_rows = []
    batch_finalist_histories = {}
    for label, folder, base_name in batch_finalists:
        folder_path = results_root / folder
        summaries = {
            row["run"]: row
            for row in json.loads((folder_path / "summary.json").read_text())
        }
        batch_finalist_histories[label] = []
        for seed in range(5):
            run_name = f"{base_name}-seed-{seed}"
            run_folder = folder_path / run_name
            history = _read_history(run_folder)
            batch_finalist_histories[label].append(history)
            best = _best(history)
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            batch_finalist_rows.append({
                "candidate": label,
                "seed": seed,
                **best,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "digit_5_f1": digit_five["f1"] or 0.0,
                "ignored_digit_5": digit_five["true_positive"] == 0,
                "updates_at_best_epoch": summaries[run_name][
                    "updates_at_best_epoch"],
                "seconds": summaries[run_name]["seconds"],
            })
    _write_csv(output / "adam-batch-finalists-seeds.csv", batch_finalist_rows)

    batch_aggregate_rows = []
    batch_labels = [item[0] for item in batch_finalists]
    for label in batch_labels:
        rows = [row for row in batch_finalist_rows if row["candidate"] == label]
        batch_aggregate_rows.append({
            "candidate": label,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "ignored_digit_5_runs": sum(row["ignored_digit_5"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
            "updates_at_best_mean": statistics.mean(
                row["updates_at_best_epoch"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "adam-batch-finalists-summary.csv", batch_aggregate_rows)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    batch_positions = list(range(len(batch_labels)))
    for position, label in enumerate(batch_labels):
        rows = [row for row in batch_finalist_rows if row["candidate"] == label]
        axes[0, 0].scatter([position] * 5,
                           [row["validation_macro_f1_present"] for row in rows])
        axes[0, 1].scatter([position] * 5,
                           [row["digit_5_f1"] for row in rows])
    axes[0, 0].boxplot([
        [row["validation_macro_f1_present"] for row in batch_finalist_rows
         if row["candidate"] == label] for label in batch_labels],
        positions=batch_positions)
    axes[0, 1].boxplot([
        [row["digit_5_f1"] for row in batch_finalist_rows
         if row["candidate"] == label] for label in batch_labels],
        positions=batch_positions)
    axes[1, 0].bar(batch_positions,
                   [row["updates_at_best_mean"] for row in batch_aggregate_rows])
    axes[1, 1].bar(batch_positions,
                   [row["seconds_mean"] for row in batch_aggregate_rows])
    axes[0, 0].set(title="Macro-F1 en cinco semillas", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5", ylabel="F1")
    axes[1, 0].set(title="Actualizaciones hasta la mejor época",
                   ylabel="Actualizaciones")
    axes[1, 1].set(title="Tiempo total medio", ylabel="Segundos")
    for axis in axes.flat:
        axis.set_xticks(batch_positions, batch_labels, rotation=15, ha="right")
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Finalistas de batch con Adam")
    figure.tight_layout()
    figure.savefig(output / "adam-batch-finalists.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for axis, label in zip(axes, batch_labels):
        histories = batch_finalist_histories[label]
        epochs = histories[0]["epoch"]
        for metric, linestyle, color in (
                ("training_macro_f1_present", "--", "tab:blue"),
                ("validation_macro_f1_present", "-", "tab:orange")):
            means = [statistics.mean(history[metric][index]
                                     for history in histories)
                     for index in range(len(epochs))]
            deviations = [statistics.stdev(history[metric][index]
                                            for history in histories)
                          for index in range(len(epochs))]
            axis.plot(epochs, means, color=color, linestyle=linestyle,
                      label="train" if metric.startswith("training") else "validation")
            axis.fill_between(
                epochs,
                [mean - deviation for mean, deviation in zip(means, deviations)],
                [mean + deviation for mean, deviation in zip(means, deviations)],
                color=color, alpha=0.15)
        axis.set(title=label, xlabel="Época", ylabel="Macro-F1", ylim=(0.15, 1.01))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Curvas medias ± un desvío de los batches finalistas")
    figure.tight_layout()
    figure.savefig(output / "adam-batch-finalist-curves.png", dpi=180)
    plt.close(figure)

    modality_pilots = (
        ("Online", 1, 0.0001, "15a-adam-online-pilot/online-lr-00001"),
        ("Online", 1, 0.0003, "15a-adam-online-pilot/online-lr-00003"),
        ("Online", 1, 0.001, "15a-adam-online-pilot/online-lr-0001"),
        ("Online", 1, 0.003, "15a-adam-online-pilot/online-lr-0003"),
        ("Full batch", None, 0.003, "15b-adam-full-batch-pilot/full-lr-0003"),
        ("Full batch", None, 0.01, "15b-adam-full-batch-pilot/full-lr-001"),
        ("Full batch", None, 0.03, "15b-adam-full-batch-pilot/full-lr-003"),
    )
    modality_summaries = {}
    for folder in ("15a-adam-online-pilot", "15b-adam-full-batch-pilot"):
        for row in json.loads((results_root / folder / "summary.json").read_text()):
            modality_summaries[row["run"]] = row
    modality_pilot_rows = []
    modality_pilot_histories = []
    for modality, batch, learning_rate, relative in modality_pilots:
        run_folder = results_root / relative
        history = _read_history(run_folder)
        best = _best(history)
        metrics = json.loads((run_folder / "metrics.json").read_text())
        digit_five = next(item for item in metrics["validation"]["per_class"]
                          if item["label"] == 5)
        run_name = relative.rsplit("/", 1)[-1]
        modality_pilot_rows.append({
            "modality": modality,
            "batch": "full" if batch is None else batch,
            "learning_rate": learning_rate,
            **best,
            "generalization_gap": (
                best["training_macro_f1_present"]
                - best["validation_macro_f1_present"]),
            "digit_5_f1": digit_five["f1"] or 0.0,
            "updates_at_best_epoch": modality_summaries[run_name][
                "updates_at_best_epoch"],
            "seconds": modality_summaries[run_name]["seconds"],
        })
        modality_pilot_histories.append(
            (modality, batch, learning_rate, history))
    _write_csv(output / "adam-extreme-modalities-pilots.csv",
               modality_pilot_rows)

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for modality, _, learning_rate, history in modality_pilot_histories:
        axis = axes[0] if modality == "Online" else axes[1]
        axis.plot(history["epoch"], history["validation_macro_f1_present"],
                  label=f"η={learning_rate:g}")
    axes[0].set(title="Online: 5 épocas", xlabel="Época",
                ylabel="Macro-F1 validation", ylim=(-0.02, 1.0))
    axes[1].set(title="Full batch: 3.000 épocas", xlabel="Época",
                ylabel="Macro-F1 validation", ylim=(-0.02, 1.0))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Learning rate específico para modalidades extremas")
    figure.tight_layout()
    figure.savefig(output / "adam-extreme-modalities-learning-rates.png", dpi=180)
    plt.close(figure)

    modality_finalists = (
        ("Online, η=0.001", "16-adam-online-finalist-seeds", "online-lr-0001"),
        ("Mini-batch 128, η=0.003",
         "12-architecture-optimizer-finalists-seeds", "arch-128-adam-lr-0003"),
    )
    modality_seed_rows = []
    modality_histories = {}
    for label, folder, base_name in modality_finalists:
        folder_path = results_root / folder
        summaries = {
            row["run"]: row
            for row in json.loads((folder_path / "summary.json").read_text())
        }
        modality_histories[label] = []
        for seed in range(5):
            run_name = f"{base_name}-seed-{seed}"
            run_folder = folder_path / run_name
            history = _read_history(run_folder)
            modality_histories[label].append(history)
            best = _best(history)
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            modality_seed_rows.append({
                "candidate": label,
                "seed": seed,
                **best,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "digit_5_f1": digit_five["f1"] or 0.0,
                "updates_at_best_epoch": summaries[run_name][
                    "updates_at_best_epoch"],
                "seconds": summaries[run_name]["seconds"],
            })
    _write_csv(output / "adam-extreme-modalities-seeds.csv", modality_seed_rows)

    modality_aggregate_rows = []
    modality_labels = [item[0] for item in modality_finalists]
    for label in modality_labels:
        rows = [row for row in modality_seed_rows if row["candidate"] == label]
        modality_aggregate_rows.append({
            "candidate": label,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
            "updates_at_best_mean": statistics.mean(
                row["updates_at_best_epoch"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "adam-extreme-modalities-summary.csv",
               modality_aggregate_rows)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    modality_positions = list(range(len(modality_labels)))
    for position, label in enumerate(modality_labels):
        rows = [row for row in modality_seed_rows if row["candidate"] == label]
        axes[0, 0].scatter([position] * 5,
                           [row["validation_macro_f1_present"] for row in rows])
        axes[0, 1].scatter([position] * 5,
                           [row["digit_5_f1"] for row in rows])
    axes[0, 0].boxplot([
        [row["validation_macro_f1_present"] for row in modality_seed_rows
         if row["candidate"] == label] for label in modality_labels],
        positions=modality_positions)
    axes[0, 1].boxplot([
        [row["digit_5_f1"] for row in modality_seed_rows
         if row["candidate"] == label] for label in modality_labels],
        positions=modality_positions)
    axes[1, 0].bar(modality_positions,
                   [row["updates_at_best_mean"] for row in modality_aggregate_rows])
    axes[1, 1].bar(modality_positions,
                   [row["seconds_mean"] for row in modality_aggregate_rows])
    axes[0, 0].set(title="Macro-F1 en cinco semillas", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5", ylabel="F1")
    axes[1, 0].set(title="Actualizaciones hasta la mejor época",
                   ylabel="Actualizaciones")
    axes[1, 1].set(title="Tiempo total medio", ylabel="Segundos")
    for axis in axes.flat:
        axis.set_xticks(modality_positions, modality_labels, rotation=10,
                        ha="right")
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Online frente a mini-batch")
    figure.tight_layout()
    figure.savefig(output / "adam-extreme-modalities-finalists.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    comparison_histories = (
        ("Online η=0.001", 1,
         _read_history(results_root / "15d-adam-online-final-extension"
                       / "online-lr-0001-final-extension")),
        ("Mini-batch 128 η=0.003", 128,
         _read_history(results_root / "12-architecture-optimizer-finalists-seeds"
                       / "arch-128-adam-lr-0003-seed-0")),
        ("Full batch η=0.003", None,
         _read_history(results_root / "15b-adam-full-batch-pilot"
                       / "full-lr-0003")),
    )
    training_size = 9959
    for label, batch, history in comparison_histories:
        updates_per_epoch = 1 if batch is None else (
            training_size + batch - 1) // batch
        updates = [epoch * updates_per_epoch for epoch in history["epoch"]]
        axis.plot(updates, history["validation_macro_f1_present"], label=label)
    axis.set_xscale("symlog", linthresh=1)
    axis.set(title="Validation según cantidad de actualizaciones (semilla 0)",
             xlabel="Actualizaciones acumuladas (escala symlog)",
             ylabel="Macro-F1", ylim=(-0.02, 1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "adam-extreme-modalities-by-updates.png", dpi=180)
    plt.close(figure)

    partition_sources = {
        0: {
            "Adam 128": ("12-architecture-optimizer-finalists-seeds",
                         "arch-128-adam-lr-0003-seed-0"),
            "RMSProp 64": ("10-optimizer-finalists-seeds",
                           "rmsprop-lr-001-seed-0"),
        },
        1: {
            "Adam 128": ("17a-partition-sensitivity-split-1", "adam-128"),
            "RMSProp 64": ("17a-partition-sensitivity-split-1", "rmsprop-64"),
        },
        2: {
            "Adam 128": ("17b-partition-sensitivity-split-2", "adam-128"),
            "RMSProp 64": ("17b-partition-sensitivity-split-2", "rmsprop-64"),
        },
        3: {
            "Adam 128": ("17c-partition-sensitivity-split-3", "adam-128"),
            "RMSProp 64": ("17c-partition-sensitivity-split-3", "rmsprop-64"),
        },
        4: {
            "Adam 128": ("17d-partition-sensitivity-split-4", "adam-128"),
            "RMSProp 64": ("17d-partition-sensitivity-split-4", "rmsprop-64"),
        },
    }
    partition_rows = []
    for split, candidates in partition_sources.items():
        for candidate, (folder, run_name) in candidates.items():
            run_folder = results_root / folder / run_name
            history = _read_history(run_folder)
            best = _best(history)
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            partition_rows.append({
                "validation_seed": split,
                "candidate": candidate,
                **best,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "digit_5_f1": digit_five["f1"] or 0.0,
            })
    _write_csv(output / "partition-sensitivity-seeds.csv", partition_rows)

    partition_candidates = ("Adam 128", "RMSProp 64")
    partition_aggregate_rows = []
    for candidate in partition_candidates:
        rows = [row for row in partition_rows if row["candidate"] == candidate]
        partition_aggregate_rows.append({
            "candidate": candidate,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
        })
    _write_csv(output / "partition-sensitivity-summary.csv",
               partition_aggregate_rows)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    split_positions = list(range(5))
    colors = {"Adam 128": "tab:blue", "RMSProp 64": "tab:orange"}
    for candidate in partition_candidates:
        rows = sorted(
            (row for row in partition_rows if row["candidate"] == candidate),
            key=lambda row: row["validation_seed"])
        axes[0, 0].plot(split_positions,
                        [row["validation_macro_f1_present"] for row in rows],
                        marker="o", color=colors[candidate], label=candidate)
        axes[0, 1].plot(split_positions,
                        [row["digit_5_f1"] for row in rows], marker="o",
                        color=colors[candidate], label=candidate)
        axes[1, 0].plot(split_positions,
                        [row["generalization_gap"] for row in rows], marker="o",
                        color=colors[candidate], label=candidate)
        axes[1, 1].plot(split_positions,
                        [row["best_epoch"] for row in rows], marker="o",
                        color=colors[candidate], label=candidate)
    axes[0, 0].set(title="Macro-F1 por partición", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5 por partición", ylabel="F1")
    axes[1, 0].set(title="Gap train − validation", ylabel="Gap")
    axes[1, 1].set(title="Mejor época", ylabel="Época")
    for axis in axes.flat:
        axis.set(xlabel="validation_seed", xticks=split_positions)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Sensibilidad a la partición training–validation")
    figure.tight_layout()
    figure.savefig(output / "partition-sensitivity.png", dpi=180)
    plt.close(figure)

    paired_adam = sorted(
        (row for row in partition_rows if row["candidate"] == "Adam 128"),
        key=lambda row: row["validation_seed"])
    paired_rmsprop = sorted(
        (row for row in partition_rows if row["candidate"] == "RMSProp 64"),
        key=lambda row: row["validation_seed"])
    macro_differences = [
        adam["validation_macro_f1_present"] - rmsprop["validation_macro_f1_present"]
        for adam, rmsprop in zip(paired_adam, paired_rmsprop)]
    digit_differences = [
        adam["digit_5_f1"] - rmsprop["digit_5_f1"]
        for adam, rmsprop in zip(paired_adam, paired_rmsprop)]
    paired_rows = [{
        "metric": "macro_f1_adam_minus_rmsprop",
        "mean_difference": statistics.mean(macro_differences),
        "sd_difference": statistics.stdev(macro_differences),
        "adam_wins": sum(value > 0 for value in macro_differences),
        "ties": sum(value == 0 for value in macro_differences),
        "rmsprop_wins": sum(value < 0 for value in macro_differences),
    }, {
        "metric": "digit_5_f1_adam_minus_rmsprop",
        "mean_difference": statistics.mean(digit_differences),
        "sd_difference": statistics.stdev(digit_differences),
        "adam_wins": sum(value > 0 for value in digit_differences),
        "ties": sum(value == 0 for value in digit_differences),
        "rmsprop_wins": sum(value < 0 for value in digit_differences),
    }]
    _write_csv(output / "partition-sensitivity-paired.csv", paired_rows)

    cross_validation_candidates = (
        ("64 + Adam η=0.01", "arch-64-adam-lr-001", 50890),
        ("64 + RMSProp η=0.01", "arch-64-rmsprop-lr-001", 50890),
        ("128 + Adam η=0.003", "arch-128-adam-lr-0003", 101770),
        ("128 + RMSProp η=0.01", "arch-128-rmsprop-lr-001", 101770),
    )
    cross_validation_folders = tuple(
        f"18{letter}-cross-validation-fold-{fold}"
        for fold, letter in enumerate("abcde"))
    cross_validation_rows = []
    for fold, folder in enumerate(cross_validation_folders):
        summaries = {
            row["run"]: row for row in json.loads(
                (results_root / folder / "summary.json").read_text())
        }
        for candidate, run_name, parameter_count in cross_validation_candidates:
            run_folder = results_root / folder / run_name
            history = _read_history(run_folder)
            best = _best(history)
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            cross_validation_rows.append({
                "fold": fold,
                "candidate": candidate,
                **best,
                "digit_5_f1": digit_five["f1"] or 0.0,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "parameter_count": parameter_count,
                "seconds": summaries[run_name]["seconds"],
            })
    _write_csv(output / "cross-validation-folds.csv", cross_validation_rows)

    cross_validation_summary = []
    for candidate, _, parameter_count in cross_validation_candidates:
        rows = [row for row in cross_validation_rows
                if row["candidate"] == candidate]
        cross_validation_summary.append({
            "candidate": candidate,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
            "parameter_count": parameter_count,
        })
    _write_csv(output / "cross-validation-summary.csv", cross_validation_summary)

    reference = "64 + RMSProp η=0.01"
    reference_rows = sorted(
        (row for row in cross_validation_rows if row["candidate"] == reference),
        key=lambda row: row["fold"])
    cross_validation_paired = []
    for candidate, _, _ in cross_validation_candidates:
        if candidate == reference:
            continue
        rows = sorted(
            (row for row in cross_validation_rows if row["candidate"] == candidate),
            key=lambda row: row["fold"])
        for metric, column in (("macro_f1", "validation_macro_f1_present"),
                               ("digit_5_f1", "digit_5_f1")):
            differences = [row[column] - baseline[column]
                           for row, baseline in zip(rows, reference_rows)]
            cross_validation_paired.append({
                "candidate_minus_reference": candidate,
                "reference": reference,
                "metric": metric,
                "mean_difference": statistics.mean(differences),
                "sd_difference": statistics.stdev(differences),
                "candidate_wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "reference_wins": sum(value < 0 for value in differences),
            })
    _write_csv(output / "cross-validation-paired.csv", cross_validation_paired)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for (candidate, _, _), color in zip(cross_validation_candidates, colors):
        rows = sorted(
            (row for row in cross_validation_rows if row["candidate"] == candidate),
            key=lambda row: row["fold"])
        axes[0, 0].plot(range(5),
                        [row["validation_macro_f1_present"] for row in rows],
                        marker="o", label=candidate, color=color)
        axes[0, 1].plot(range(5), [row["digit_5_f1"] for row in rows],
                        marker="o", label=candidate, color=color)
        axes[1, 0].plot(range(5), [row["generalization_gap"] for row in rows],
                        marker="o", label=candidate, color=color)
        axes[1, 1].plot(range(5), [row["best_epoch"] for row in rows],
                        marker="o", label=candidate, color=color)
    axes[0, 0].set(title="Macro-F1 por fold", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5 por fold", ylabel="F1")
    axes[1, 0].set(title="Gap train − validation", ylabel="Gap")
    axes[1, 1].set(title="Mejor época", ylabel="Época")
    for axis in axes.flat:
        axis.set(xlabel="Fold", xticks=range(5))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Validación cruzada estratificada de cinco folds")
    figure.tight_layout()
    figure.savefig(output / "cross-validation-folds.png", dpi=180)
    plt.close(figure)

    labels = [row["candidate"] for row in cross_validation_summary]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].errorbar(
        positions,
        [row["macro_f1_mean"] for row in cross_validation_summary],
        yerr=[row["macro_f1_sd"] for row in cross_validation_summary],
        fmt="o", capsize=5)
    axes[0, 1].errorbar(
        positions,
        [row["digit_5_f1_mean"] for row in cross_validation_summary],
        yerr=[row["digit_5_f1_sd"] for row in cross_validation_summary],
        fmt="o", capsize=5)
    axes[1, 0].bar(positions,
                   [row["seconds_mean"] for row in cross_validation_summary])
    axes[1, 1].bar(positions,
                   [row["parameter_count"] for row in cross_validation_summary])
    axes[0, 0].set(title="Macro-F1 medio ± sd", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5 medio ± sd", ylabel="F1")
    axes[1, 0].set(title="Tiempo medio por fold", ylabel="Segundos")
    axes[1, 1].set(title="Cantidad de parámetros", ylabel="Parámetros")
    for axis in axes.flat:
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Resumen de los cuatro candidatos en 5-fold")
    figure.tight_layout()
    figure.savefig(output / "cross-validation-summary.png", dpi=180)
    plt.close(figure)

    rmsprop_batch_root = results_root / "19-rmsprop-batch-learning-rate"
    rmsprop_batch_summaries = {
        row["run"]: row for row in json.loads(
            (rmsprop_batch_root / "summary.json").read_text())
    }
    rmsprop_batch_rows = []
    for batch in (32, 64, 128, 256, 512):
        for learning_rate, suffix in ((0.003, "0003"), (0.01, "001"),
                                      (0.03, "003")):
            run_name = f"batch-{batch}-lr-{suffix}"
            run_folder = rmsprop_batch_root / run_name
            best = _at_epoch(
                _read_history(run_folder),
                int(rmsprop_batch_summaries[run_name]["best_epoch"]))
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            rmsprop_batch_rows.append({
                "batch_size": batch,
                "learning_rate": learning_rate,
                **best,
                "digit_5_f1": digit_five["f1"] or 0.0,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "seconds": rmsprop_batch_summaries[run_name]["seconds"],
            })
    _write_csv(output / "rmsprop-batch-learning-rate.csv", rmsprop_batch_rows)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for learning_rate in (0.003, 0.01, 0.03):
        rows = [row for row in rmsprop_batch_rows
                if row["learning_rate"] == learning_rate]
        label = f"η={learning_rate:g}"
        axes[0, 0].plot(
            [row["batch_size"] for row in rows],
            [row["validation_macro_f1_present"] for row in rows],
            marker="o", label=label)
        axes[0, 1].plot(
            [row["batch_size"] for row in rows],
            [row["digit_5_f1"] for row in rows], marker="o", label=label)
        axes[1, 0].plot(
            [row["batch_size"] for row in rows],
            [row["best_epoch"] for row in rows], marker="o", label=label)
        axes[1, 1].plot(
            [row["batch_size"] for row in rows],
            [row["seconds"] for row in rows], marker="o", label=label)
    axes[0, 0].set(title="Macro-F1 en la mejor época", ylabel="Macro-F1")
    axes[0, 1].set(title="F1 del dígito 5", ylabel="F1")
    axes[1, 0].set(title="Mejor época", ylabel="Época")
    axes[1, 1].set(title="Tiempo total", ylabel="Segundos")
    for axis in axes.flat:
        axis.set(xlabel="Batch", xscale="log", xticks=(32, 64, 128, 256, 512))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("RMSProp: interacción entre batch y learning rate (split fijo)")
    figure.tight_layout()
    figure.savefig(output / "rmsprop-batch-learning-rate.png", dpi=180)
    plt.close(figure)

    rmsprop_batch_finalist_rows = []
    for fold, letter in enumerate("abcde"):
        folder = results_root / f"20{letter}-rmsprop-batch-finalists-fold-{fold}"
        summaries = {
            row["run"]: row for row in json.loads(
                (folder / "summary.json").read_text())
        }
        for batch in (32, 64):
            run_name = f"batch-{batch}-lr-001"
            run_folder = folder / run_name
            best = _at_epoch(
                _read_history(run_folder), int(summaries[run_name]["best_epoch"]))
            metrics = json.loads((run_folder / "metrics.json").read_text())
            digit_five = next(
                item for item in metrics["validation"]["per_class"]
                if item["label"] == 5)
            rmsprop_batch_finalist_rows.append({
                "fold": fold,
                "candidate": f"RMSProp-64 batch {batch}",
                **best,
                "digit_5_f1": digit_five["f1"] or 0.0,
                "generalization_gap": (
                    best["training_macro_f1_present"]
                    - best["validation_macro_f1_present"]),
                "seconds": summaries[run_name]["seconds"],
            })
    for row in cross_validation_rows:
        if row["candidate"] == "64 + RMSProp η=0.01":
            copied = dict(row)
            copied.pop("parameter_count", None)
            copied["candidate"] = "RMSProp-64 batch 128"
            rmsprop_batch_finalist_rows.append(copied)
        elif row["candidate"] == "128 + Adam η=0.003":
            copied = dict(row)
            copied.pop("parameter_count", None)
            copied["candidate"] = "Adam-128 batch 128"
            rmsprop_batch_finalist_rows.append(copied)
    _write_csv(output / "rmsprop-batch-finalists-folds.csv",
               rmsprop_batch_finalist_rows)

    rmsprop_batch_finalist_labels = (
        "RMSProp-64 batch 32", "RMSProp-64 batch 64",
        "RMSProp-64 batch 128", "Adam-128 batch 128")
    rmsprop_batch_finalist_summary = []
    for candidate in rmsprop_batch_finalist_labels:
        rows = [row for row in rmsprop_batch_finalist_rows
                if row["candidate"] == candidate]
        rmsprop_batch_finalist_summary.append({
            "candidate": candidate,
            "macro_f1_mean": statistics.mean(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_sd": statistics.stdev(
                row["validation_macro_f1_present"] for row in rows),
            "macro_f1_min": min(
                row["validation_macro_f1_present"] for row in rows),
            "digit_5_f1_mean": statistics.mean(row["digit_5_f1"] for row in rows),
            "digit_5_f1_sd": statistics.stdev(row["digit_5_f1"] for row in rows),
            "gap_mean": statistics.mean(row["generalization_gap"] for row in rows),
            "best_epoch_mean": statistics.mean(row["best_epoch"] for row in rows),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
        })
    _write_csv(output / "rmsprop-batch-finalists-summary.csv",
               rmsprop_batch_finalist_summary)

    baseline_rows = sorted(
        (row for row in rmsprop_batch_finalist_rows
         if row["candidate"] == "RMSProp-64 batch 128"),
        key=lambda row: row["fold"])
    rmsprop_batch_paired = []
    for candidate in rmsprop_batch_finalist_labels:
        if candidate == "RMSProp-64 batch 128":
            continue
        rows = sorted(
            (row for row in rmsprop_batch_finalist_rows
             if row["candidate"] == candidate), key=lambda row: row["fold"])
        for metric, column in (("macro_f1", "validation_macro_f1_present"),
                               ("digit_5_f1", "digit_5_f1")):
            differences = [row[column] - baseline[column]
                           for row, baseline in zip(rows, baseline_rows)]
            rmsprop_batch_paired.append({
                "candidate_minus_batch_128": candidate,
                "metric": metric,
                "mean_difference": statistics.mean(differences),
                "sd_difference": statistics.stdev(differences),
                "candidate_wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "batch_128_wins": sum(value < 0 for value in differences),
            })
    _write_csv(output / "rmsprop-batch-finalists-paired.csv",
               rmsprop_batch_paired)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for candidate, color in zip(rmsprop_batch_finalist_labels, colors):
        rows = sorted(
            (row for row in rmsprop_batch_finalist_rows
             if row["candidate"] == candidate), key=lambda row: row["fold"])
        axes[0, 0].plot(
            range(5), [row["validation_macro_f1_present"] for row in rows],
            marker="o", label=candidate, color=color)
        axes[0, 1].plot(
            range(5), [row["digit_5_f1"] for row in rows],
            marker="o", label=candidate, color=color)
    positions = list(range(len(rmsprop_batch_finalist_labels)))
    axes[1, 0].bar(positions, [row["seconds_mean"]
                              for row in rmsprop_batch_finalist_summary])
    axes[1, 1].bar(positions, [row["best_epoch_mean"]
                              for row in rmsprop_batch_finalist_summary])
    axes[0, 0].set(title="Macro-F1 por fold", xlabel="Fold", ylabel="Macro-F1",
                   xticks=range(5))
    axes[0, 1].set(title="F1 del dígito 5 por fold", xlabel="Fold", ylabel="F1",
                   xticks=range(5))
    axes[1, 0].set(title="Tiempo medio", ylabel="Segundos")
    axes[1, 1].set(title="Mejor época media", ylabel="Época")
    for axis in axes[:1].flat:
        axis.legend(fontsize=8)
    for axis in axes[1].flat:
        axis.set_xticks(positions, rmsprop_batch_finalist_labels,
                        rotation=12, ha="right")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("Confirmación 5-fold de batch con RMSProp")
    figure.tight_layout()
    figure.savefig(output / "rmsprop-batch-finalists-cross-validation.png",
                   dpi=180)
    plt.close(figure)

    final_root = results_root / "21-final-adam-128"
    final_metrics = json.loads((final_root / "metrics.json").read_text())
    final_summary = json.loads((final_root / "summary.json").read_text())
    test_per_class = final_metrics["test"]["per_class"]
    non_eight = [row for row in test_per_class if row["label"] != 8]
    confusion = final_metrics["test"]["confusion_matrix"]
    non_eight_samples = sum(row["support"] for row in non_eight)
    non_eight_correct = sum(
        confusion[label][label] for label in range(10) if label != 8)
    final_summary_row = {
        **final_summary,
        "test_macro_f1_without_8": statistics.mean(
            row["f1"] for row in non_eight),
        "test_accuracy_without_8": non_eight_correct / non_eight_samples,
        "digit_8_support": test_per_class[8]["support"],
        "digit_8_predicted": test_per_class[8]["predicted"],
        "digit_8_f1": test_per_class[8]["f1"],
    }
    _write_csv(output / "final-test-summary.csv", [final_summary_row])
    _write_csv(output / "final-test-per-class.csv", test_per_class)

    with (final_root / "training-history.csv").open(newline="") as file:
        final_history_rows = list(csv.DictReader(file))
    final_epochs = [int(row["epoch"]) for row in final_history_rows]
    final_training_mse = [float(row["mse"]) for row in final_history_rows]
    final_training_f1 = [
        float(row["training_macro_f1_present"]) for row in final_history_rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(final_epochs, final_training_mse)
    axes[0].set(title="MSE sobre development completo", xlabel="Época",
                ylabel="MSE", yscale="log")
    axes[1].plot(final_epochs, final_training_f1)
    axes[1].set(title="Macro-F1 de training (clases presentes)",
                xlabel="Época", ylabel="Macro-F1", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Entrenamiento final congelado: Adam-128, 200 épocas")
    figure.tight_layout()
    figure.savefig(output / "final-training-curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    image = axes[0].imshow(confusion, cmap="Blues")
    for expected, row in enumerate(confusion):
        for predicted, value in enumerate(row):
            if value:
                axes[0].text(predicted, expected, str(value), ha="center",
                             va="center", fontsize=8,
                             color="white" if value > 140 else "black")
    axes[0].set(title="Matriz de confusión en test", xlabel="Predicción",
                ylabel="Clase real", xticks=range(10), yticks=range(10))
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    labels = [row["label"] for row in test_per_class]
    f1_values = [row["f1"] for row in test_per_class]
    bars = axes[1].bar(labels, f1_values)
    bars[8].set_color("tab:red")
    axes[1].set(title="F1 por dígito en test", xlabel="Dígito", ylabel="F1",
                xticks=range(10), ylim=(0, 1.02))
    axes[1].axhline(final_summary["test_macro_f1"], linestyle="--",
                    color="black", linewidth=1,
                    label=f"Macro-F1 global = {final_summary['test_macro_f1']:.3f}")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()
    figure.suptitle("Evaluación final única sobre digits_test.csv")
    figure.tight_layout()
    figure.savefig(output / "final-test-confusion-and-f1.png", dpi=180)
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

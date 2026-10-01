"""Estudiar convergencia y saturación del perceptrón logístico de fraude."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3.experiments import load_fraud_learning_data
from sia_tp3.models import Perceptron
from sia_tp3.optimizers import (Adam, AdaptiveLearningRate, GradientDescent,
                                Momentum, RMSProp)
from sia_tp3.training import fit


EXPERIMENT = {
    "activation": "logistic",
    "beta": 1.0,
    "init_scale": 0.5,
    "shuffle": True,
    "target_mse": 0.0,
    "learning_rates": [0.0001, 0.001, 0.01, 0.1],
    "optimizer_learning_rates": {
        "gradient_descent": [0.001, 0.01, 0.1],
        "momentum": [0.001, 0.01, 0.1],
        "adaptive": [0.001, 0.01, 0.1],
        "rmsprop": [0.0001, 0.001, 0.01],
        "adam": [0.0001, 0.001, 0.01],
    },
    "optimizer_parameters": {
        "momentum": {"alpha": 0.9},
        "adaptive": {"increase_by": 0.0001, "decrease_fraction": 0.5,
                     "patience": 10},
        "rmsprop": {"gamma": 0.9, "epsilon": 1e-8},
        "adam": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    },
    "optimizer_screen_epochs": 300,
    "optimizer_screen_batch_size": 32,
    "batch_sizes": [1, 8, 32, 128, 512, None],
    "batch_epochs": 300,
    "extended_batch_epochs": {"128": 1000, "512": 2000, "full": 10000},
    "seeds": [0, 1, 2, 3, 4],
    "seed_epochs": 500,
    "long_run_epochs": 1000,
    "reported_epoch_checkpoints": [0, 10, 25, 50, 100, 250, 500, 750, 1000],
}


def make_optimizer(name, learning_rate):
    """Construir un optimizador con los parámetros declarados del estudio."""
    parameters = EXPERIMENT["optimizer_parameters"]
    if name == "gradient_descent":
        return GradientDescent(learning_rate)
    if name == "momentum":
        return Momentum(learning_rate, **parameters[name])
    if name == "adaptive":
        return AdaptiveLearningRate(learning_rate, **parameters[name])
    if name == "rmsprop":
        return RMSProp(learning_rate, **parameters[name])
    if name == "adam":
        return Adam(learning_rate, **parameters[name])
    raise ValueError(f"optimizador desconocido: {name}")


def plateau_epoch(mse, tolerance=0.01):
    """Primera época tras la cual el MSE queda dentro de 1 % del mínimo."""
    values = np.asarray(mse, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("mse debe ser una serie finita no vacía")
    limit = values.min() * (1.0 + tolerance)
    suffix_max = np.maximum.accumulate(values[::-1])[::-1]
    candidates = np.flatnonzero(suffix_max <= limit)
    return int(candidates[0]) if len(candidates) else None


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no hay filas para escribir en {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train_case(data, *, phase, case, optimizer_name, learning_rate,
               batch_size, max_epochs, model_seed, shuffle_seed):
    """Ejecutar una corrida de aprendizaje sobre las 7500 muestras."""
    model = Perceptron(
        data.X.shape[1], activation=EXPERIMENT["activation"],
        beta=EXPERIMENT["beta"], seed=model_seed,
        init_scale=EXPERIMENT["init_scale"],
    )
    optimizer = make_optimizer(optimizer_name, learning_rate)
    started = perf_counter()
    try:
        history = fit(
            model, data.X, data.y, optimizer=optimizer,
            batch_size=batch_size, max_epochs=max_epochs,
            target_mse=EXPERIMENT["target_mse"],
            shuffle=EXPERIMENT["shuffle"], seed=shuffle_seed,
        )
        status = "ok"
        error = ""
    except FloatingPointError as caught:
        history = []
        status = "diverged"
        error = str(caught)
    seconds = perf_counter() - started
    batch_label = "full" if batch_size is None else str(batch_size)
    identity = {
        "phase": phase,
        "case": case,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_label,
        "max_epochs": max_epochs,
        "model_seed": model_seed,
        "shuffle_seed": shuffle_seed,
    }
    if not history:
        return ({**identity, "status": status, "error": error,
                 "seconds": seconds, "final_learning_rate": optimizer.learning_rate,
                 "final_mse": "", "minimum_mse": "", "minimum_epoch": "",
                 "plateau_epoch_1pct": "", "tail_range_last_20pct": ""}, [])
    mse = np.asarray([row["mse"] for row in history])
    tail_start = int(np.floor(0.8 * (len(mse) - 1)))
    summary = {
        **identity, "status": status, "error": error, "seconds": seconds,
        "final_learning_rate": optimizer.learning_rate,
        "final_mse": float(mse[-1]), "minimum_mse": float(mse.min()),
        "minimum_epoch": int(np.argmin(mse)),
        "plateau_epoch_1pct": plateau_epoch(mse),
        "tail_range_last_20pct": float(np.ptp(mse[tail_start:])),
    }
    rows = [{**identity, "epoch": int(row["epoch"]), "mse": float(row["mse"])}
            for row in history]
    return summary, rows


def _run(data, summaries, histories, **case):
    print(f"{case['phase']}: {case['case']}", flush=True)
    summary, history = train_case(data, **case)
    summaries.append(summary)
    histories.extend(history)


def _series(histories, phase):
    selected = [row for row in histories if row["phase"] == phase]
    groups = {}
    for row in selected:
        groups.setdefault(row["case"], [[], []])
        groups[row["case"]][0].append(row["epoch"])
        groups[row["case"]][1].append(row["mse"])
    return {key: (np.asarray(value[0]), np.asarray(value[1]))
            for key, value in groups.items()}


def plot_curves(histories, phase, title, output, *, logarithmic=True):
    series = _series(histories, phase)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for label, (epochs, mse) in series.items():
        ax.plot(epochs, mse, linewidth=1.8, label=label)
    if logarithmic:
        ax.set_yscale("log")
    ax.set(title=title, xlabel="Época", ylabel="MSE de training")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_final_mse(summaries, phase, title, output):
    rows = [row for row in summaries if row["phase"] == phase and row["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    x = np.arange(len(rows))
    ax.bar(x, [row["minimum_mse"] for row in rows], color="#2878a0")
    ax.set_xticks(x, [row["case"] for row in rows], rotation=35, ha="right")
    ax.set(title=title, ylabel="Menor MSE de training")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_pc1_models(data, output):
    """Mostrar un corte 1D de los dos perceptrones entrenados sobre datos reales."""
    _, singular_values, vh = np.linalg.svd(data.X, full_matrices=False)
    pc1 = vh[0]
    scores = np.sum(data.X * pc1, axis=1)
    targets = data.y[:, 0]
    if np.corrcoef(scores, targets)[0, 1] < 0:
        pc1 = -pc1
        scores = -scores
    explained = float(singular_values[0] ** 2 / np.sum(singular_values ** 2))
    grid = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 500)
    cross_section = np.outer(grid, pc1)
    curves = {}
    for activation in ("linear", "logistic"):
        model = Perceptron(data.X.shape[1], activation=activation, beta=1.0,
                           seed=0, init_scale=0.5)
        fit(model, data.X, data.y, optimizer=GradientDescent(0.01), batch_size=32,
            max_epochs=500, target_mse=0.0, shuffle=True, seed=0)
        h = np.sum(cross_section * model.weights[0][0], axis=1) + model.biases[0][0]
        curves[activation] = (h if activation == "linear" else
                              np.exp(-np.logaddexp(0.0, -2.0 * h)))
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    ax.scatter(scores, targets, s=8, alpha=0.09, color="#64748b", linewidths=0,
               label="Probabilidad real de BigModel")
    ax.plot(grid, curves["linear"], color="#e64b35", linewidth=3,
            label="Perceptrón lineal")
    ax.plot(grid, curves["logistic"], color="#2389da", linewidth=3,
            label="Perceptrón logístico")
    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle=":")
    ax.axhline(1, color="#94a3b8", linewidth=0.8, linestyle=":")
    ax.set(
        title="Datos y modelos sobre PC1\nCorte con las demás componentes en su media",
        xlabel=f"PC1 de las 9 entradas estandarizadas ({explained:.1%} de la varianza)",
        ylabel="Probabilidad de fraude / salida del modelo",
    )
    ax.set_ylim(min(-0.1, curves["linear"].min() - 0.05),
                max(1.1, curves["linear"].max() + 0.05))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return {
        "pc1_explained_variance_fraction": explained,
        "pc1_target_correlation": float(np.corrcoef(scores, targets)[0, 1]),
        "constant_baseline_mse": float(np.mean((targets - targets.mean()) ** 2)),
        "target_mean": float(targets.mean()),
    }


def run(data_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output debe ser una carpeta nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    data = load_fraud_learning_data(data_path)
    summaries, histories = [], []

    # 1. Learning rate: reproduce y amplía el control ya usado para elegir eta.
    for rate in EXPERIMENT["learning_rates"]:
        _run(data, summaries, histories, phase="learning_rate", case=f"eta={rate:g}",
             optimizer_name="gradient_descent", learning_rate=rate,
             batch_size=32, max_epochs=500, model_seed=0, shuffle_seed=0)

    # 2. Optimizadores: todos con igual batch/semilla y una grilla propia de eta.
    for optimizer_name, rates in EXPERIMENT["optimizer_learning_rates"].items():
        for rate in rates:
            _run(data, summaries, histories, phase="optimizer_screen",
                 case=f"{optimizer_name} eta={rate:g}",
                 optimizer_name=optimizer_name, learning_rate=rate,
                 batch_size=EXPERIMENT["optimizer_screen_batch_size"],
                 max_epochs=EXPERIMENT["optimizer_screen_epochs"],
                 model_seed=0, shuffle_seed=0)

    optimizer_rows = [row for row in summaries
                      if row["phase"] == "optimizer_screen" and row["status"] == "ok"]
    best_by_optimizer = {}
    for row in optimizer_rows:
        current = best_by_optimizer.get(row["optimizer"])
        if current is None or row["minimum_mse"] < current["minimum_mse"]:
            best_by_optimizer[row["optimizer"]] = row
    for optimizer_name, best in best_by_optimizer.items():
        _run(data, summaries, histories, phase="optimizer_best",
             case=f"{optimizer_name} eta={best['learning_rate']:g}",
             optimizer_name=optimizer_name, learning_rate=best["learning_rate"],
             batch_size=32, max_epochs=500, model_seed=0, shuffle_seed=0)

    # 3. Batch: mismo GD, eta y semilla. None significa batch completo.
    for batch_size in EXPERIMENT["batch_sizes"]:
        label = "full" if batch_size is None else str(batch_size)
        _run(data, summaries, histories, phase="batch", case=f"batch={label}",
             optimizer_name="gradient_descent", learning_rate=0.01,
             batch_size=batch_size, max_epochs=EXPERIMENT["batch_epochs"],
             model_seed=0, shuffle_seed=0)

    # Los lotes grandes realizan muchas menos actualizaciones por época. Se
    # extienden por separado para no confundir convergencia lenta con otro piso.
    for label, epochs in EXPERIMENT["extended_batch_epochs"].items():
        batch_size = None if label == "full" else int(label)
        _run(data, summaries, histories, phase="batch_extended",
             case=f"batch={label} ({epochs} épocas)",
             optimizer_name="gradient_descent", learning_rate=0.01,
             batch_size=batch_size, max_epochs=epochs,
             model_seed=0, shuffle_seed=0)

    # 4. Semillas: cambia a la vez inicio de pesos y orden de mezcla.
    for seed in EXPERIMENT["seeds"]:
        _run(data, summaries, histories, phase="seed", case=f"semilla={seed}",
             optimizer_name="gradient_descent", learning_rate=0.01,
             batch_size=32, max_epochs=EXPERIMENT["seed_epochs"],
             model_seed=seed, shuffle_seed=seed)

    # 5. Épocas: una única corrida larga permite leer todos los cortes sin
    # repetir entrenamiento ni cambiar aleatoriamente la trayectoria.
    _run(data, summaries, histories, phase="epochs", case="GD eta=0.01 batch=32",
         optimizer_name="gradient_descent", learning_rate=0.01,
         batch_size=32, max_epochs=EXPERIMENT["long_run_epochs"],
         model_seed=0, shuffle_seed=0)

    checkpoints = []
    long_history = [row for row in histories if row["phase"] == "epochs"]
    by_epoch = {row["epoch"]: row["mse"] for row in long_history}
    for epoch in EXPERIMENT["reported_epoch_checkpoints"]:
        checkpoints.append({"epoch": epoch, "mse": by_epoch[epoch]})

    write_csv(output / "runs.csv", summaries)
    write_csv(output / "histories.csv", histories)
    write_csv(output / "epoch-checkpoints.csv", checkpoints)
    (output / "experiment.json").write_text(
        json.dumps(EXPERIMENT, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    plot_curves(histories, "learning_rate", "Learning rates · GD · batch 32",
                output / "learning-rates.png")
    plot_curves(histories, "optimizer_best", "Mejor learning rate por optimizador",
                output / "optimizers.png")
    plot_curves(histories, "batch", "Tamaños de batch · GD · η=0,01",
                output / "batch-sizes.png")
    plot_curves(histories, "batch_extended", "Lotes grandes con más épocas · GD · η=0,01",
                output / "batch-extended.png")
    plot_curves(histories, "seed", "Sensibilidad a semillas · GD · η=0,01 · batch 32",
                output / "seeds.png")
    plot_curves(histories, "epochs", "Corrida extendida hasta 1000 épocas",
                output / "epochs.png", logarithmic=False)
    plot_final_mse(summaries, "optimizer_best", "Piso alcanzado por cada optimizador",
                   output / "optimizer-minima.png")
    derived = plot_pc1_models(data, output / "pc1-models.png")
    (output / "derived-values.json").write_text(
        json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Resultados guardados en {output}", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fraud_dataset.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.data, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

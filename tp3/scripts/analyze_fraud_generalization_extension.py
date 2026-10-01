"""Seleccionar y evaluar el perceptrón logístico sin usar test para decidir."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3.data import (FRAUD_FEATURES, _load_fraud_file,
                          _stratified_indices)
from sia_tp3.models import Perceptron
from sia_tp3.metrics import classification_metrics
from sia_tp3.optimizers import (Adam, AdaptiveLearningRate, GradientDescent,
                                Momentum, RMSProp)
from sia_tp3.preprocessing import Standardizer
from sia_tp3.training import fit


EXPERIMENT = {
    "test_fraction": 0.2,
    "test_seed": 0,
    "folds": 5,
    "fold_seed": 0,
    "activation": "logistic",
    "beta": 1.0,
    "init_scale": 0.5,
    "screen_epochs": 300,
    "batch_size": 32,
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
    "optimizer_finalists": 2,
    "batch_sizes": [1, 8, 32, 128, 512, None],
    # Los horizontes compensan, de forma aproximada, que una época online hace
    # 4800 actualizaciones y una época full batch hace sólo una por fold.
    "batch_epochs": {"1": 60, "8": 100, "32": 300, "128": 600,
                     "512": 1200, "full": 3000},
    "configuration_finalists": 5,
    "seeds": [0, 1, 2, 3, 4],
    "plateau_tolerance": 0.001,
    "coarse_threshold_start": 0.01,
    "coarse_threshold_stop": 0.99,
    "coarse_threshold_step": 0.005,
    "fine_threshold_radius": 0.03,
    "fine_threshold_step": 0.001,
}


def make_optimizer(name, learning_rate):
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


def stable_epoch(values, tolerance=0.001):
    """Primera época tras la cual la serie no supera mínimo * (1+tolerancia)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values debe ser una serie finita no vacía")
    limit = values.min() * (1.0 + tolerance)
    suffix_max = np.maximum.accumulate(values[::-1])[::-1]
    candidates = np.flatnonzero(suffix_max <= limit)
    return int(candidates[0]) if len(candidates) else len(values) - 1


def stratified_folds(labels, indices, *, folds, seed):
    """Crear folds exhaustivos y disjuntos preservando cada clase."""
    labels = np.asarray(labels)
    indices = np.asarray(indices, dtype=np.int64)
    if folds < 2 or len(indices) < folds:
        raise ValueError("se necesitan al menos dos folds con muestras")
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(folds)]
    for label in np.unique(labels[indices]):
        class_indices = rng.permutation(indices[labels[indices] == label])
        for fold, chunk in enumerate(np.array_split(class_indices, folds)):
            parts[fold].extend(chunk.tolist())
    return [rng.permutation(np.asarray(part, dtype=np.int64)) for part in parts]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no hay filas para escribir en {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _label(optimizer, learning_rate, batch_size):
    batch = "full" if batch_size is None else str(batch_size)
    return f"{optimizer} · eta={learning_rate:g} · batch={batch}"


def evaluate_cv(X_raw, y, folds, *, stage, optimizer_name, learning_rate,
                batch_size, seed, max_epochs):
    """Evaluar una configuración: cada fold ajusta su propio estandarizador."""
    histories = []
    fold_rows = []
    all_dev = np.concatenate(folds)
    started = perf_counter()
    for fold_number, validation_indices in enumerate(folds, start=1):
        training_indices = np.setdiff1d(all_dev, validation_indices,
                                        assume_unique=True)
        standardizer = Standardizer.fit(X_raw[training_indices], FRAUD_FEATURES)
        X_train = standardizer.transform(X_raw[training_indices])
        X_validation = standardizer.transform(X_raw[validation_indices])
        model = Perceptron(
            len(FRAUD_FEATURES), activation="logistic", beta=EXPERIMENT["beta"],
            init_scale=EXPERIMENT["init_scale"], seed=seed,
        )
        try:
            # Algunas implementaciones BLAS emiten warnings intermedios aun
            # cuando el resultado queda finito. El control real está en fit:
            # si parámetros, predicciones o MSE dejan de ser finitos, aborta.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                history = fit(
                    model, X_train, y[training_indices],
                    optimizer=make_optimizer(optimizer_name, learning_rate),
                    batch_size=batch_size, max_epochs=max_epochs, target_mse=0.0,
                    shuffle=True, seed=seed,
                    validation_data=(X_validation, y[validation_indices]),
                )
        except FloatingPointError:
            return None, [], []
        histories.append(history)
        fold_rows.append({
            "stage": stage,
            "configuration": _label(optimizer_name, learning_rate, batch_size),
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "batch_size": "full" if batch_size is None else batch_size,
            "seed": seed,
            "fold": fold_number,
            "training_mse": history[-1]["mse"],
            "validation_mse": history[-1]["validation_mse"],
        })

    curves = []
    for epoch in range(max_epochs + 1):
        training = np.asarray([history[epoch]["mse"] for history in histories])
        validation = np.asarray(
            [history[epoch]["validation_mse"] for history in histories])
        curves.append({
            "stage": stage,
            "configuration": _label(optimizer_name, learning_rate, batch_size),
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "batch_size": "full" if batch_size is None else batch_size,
            "seed": seed,
            "epoch": epoch,
            "training_mean": float(training.mean()),
            "training_std": float(training.std(ddof=0)),
            "validation_mean": float(validation.mean()),
            "validation_std": float(validation.std(ddof=0)),
        })
    validation_curve = np.asarray([row["validation_mean"] for row in curves])
    chosen_epoch = stable_epoch(
        validation_curve, tolerance=EXPERIMENT["plateau_tolerance"])
    chosen = curves[chosen_epoch]
    tail = validation_curve[int(0.8 * max_epochs):]
    summary = {
        "stage": stage,
        "configuration": _label(optimizer_name, learning_rate, batch_size),
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": "full" if batch_size is None else batch_size,
        "seed": seed,
        "epochs_run": max_epochs,
        "selected_epoch": chosen_epoch,
        "training_mse": chosen["training_mean"],
        "validation_mse": chosen["validation_mean"],
        "validation_std": chosen["validation_std"],
        "minimum_validation_mse": float(validation_curve.min()),
        "minimum_epoch": int(validation_curve.argmin()),
        "tail_range": float(np.ptp(tail)),
        "seconds": perf_counter() - started,
    }
    summary["estimated_seconds_to_selected_epoch"] = (
        summary["seconds"] * chosen_epoch / max_epochs)
    return summary, curves, fold_rows


def _run_cv(X, y, folds, summaries, curves, fold_details, **configuration):
    print(f"{configuration['stage']}: "
          f"{_label(configuration['optimizer_name'], configuration['learning_rate'], configuration['batch_size'])} "
          f"seed={configuration['seed']}", flush=True)
    summary, curve_rows, fold_rows = evaluate_cv(X, y, folds, **configuration)
    if summary is not None:
        summaries.append(summary)
        curves.extend(curve_rows)
        fold_details.extend(fold_rows)


def _rank(rows):
    return sorted(rows, key=lambda row: (
        row["validation_mse"], row["validation_std"], row["tail_range"],
        row["seconds"], row["configuration"],
    ))


def _aggregate_finalists(rows):
    groups = {}
    for row in rows:
        key = (row["optimizer"], row["learning_rate"], row["batch_size"])
        groups.setdefault(key, []).append(row)
    aggregated = []
    for (optimizer, rate, batch), group in groups.items():
        scores = np.asarray([row["validation_mse"] for row in group])
        fold_stds = np.asarray([row["validation_std"] for row in group])
        aggregated.append({
            "configuration": group[0]["configuration"],
            "optimizer": optimizer,
            "learning_rate": rate,
            "batch_size": batch,
            "seeds": len(group),
            "selected_epoch": int(round(np.median(
                [row["selected_epoch"] for row in group]))),
            "validation_mse": float(scores.mean()),
            "validation_seed_std": float(scores.std(ddof=0)),
            "validation_fold_std_mean": float(fold_stds.mean()),
            "tail_range_mean": float(np.mean([row["tail_range"] for row in group])),
            "seconds_mean": float(np.mean([row["seconds"] for row in group])),
            "estimated_seconds_to_selected_epoch": float(np.mean([
                row["estimated_seconds_to_selected_epoch"] for row in group
            ])),
        })
    return sorted(aggregated, key=lambda row: (
        row["validation_mse"], row["validation_seed_std"],
        row["validation_fold_std_mean"], row["tail_range_mean"],
        row["seconds_mean"], row["configuration"],
    ))


def _plot_curves(curve_rows, configurations, output, title, *, stages):
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for configuration in configurations:
        rows = [row for row in curve_rows
                if row.get("configuration") == configuration
                and row["stage"] in stages]
        by_epoch = {}
        for row in rows:
            by_epoch.setdefault(row["epoch"], []).append(row["validation_mean"])
        epochs = np.asarray(sorted(by_epoch))
        values = np.asarray([np.mean(by_epoch[epoch]) for epoch in epochs])
        ax.plot(epochs, values, linewidth=1.8, label=configuration)
    ax.set(title=title, xlabel="Época", ylabel="MSE medio de validation")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _classification_row(labels, probabilities, threshold, *, seed):
    predicted = (np.asarray(probabilities) >= threshold).astype(np.int64)
    report = classification_metrics(labels, predicted, labels=(0, 1))
    positive = report.for_label(1)
    return {
        "seed": seed,
        "threshold": float(threshold),
        "accuracy": report.accuracy,
        "precision": positive.precision,
        "recall": positive.recall,
        "f1": positive.f1,
        "tpr": positive.tpr,
        "fpr": positive.fpr,
        "true_negative": positive.true_negative,
        "false_positive": positive.false_positive,
        "false_negative": positive.false_negative,
        "true_positive": positive.true_positive,
    }


def evaluate_thresholds(labels, probabilities, thresholds, *, seed):
    """Evaluar una grilla ya fijada sin modificar probabilidades ni etiquetas."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("labels y probabilities deben ser vectores no vacíos iguales")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("clasificación binaria requiere labels 0/1 y probabilidades finitas")
    return [_classification_row(labels, probabilities, threshold, seed=seed)
            for threshold in thresholds]


def choose_threshold(rows, *, metric="f1"):
    """Maximizar una métrica; desempatar por FPR, recall y menor umbral."""
    if not rows or metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("filas no vacías y métrica de selección conocida requeridas")
    return max(rows, key=lambda row: (
        row[metric], -row["fpr"], row["recall"], -row["threshold"]
    ))


def _aggregate_thresholds(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["threshold"], []).append(row)
    result = []
    metric_names = ("accuracy", "precision", "recall", "f1", "tpr", "fpr")
    for threshold in sorted(groups):
        group = groups[threshold]
        aggregate = {"threshold": threshold, "seeds": len(group)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group])
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=0))
        result.append(aggregate)
    return result


def _plot_thresholds(rows, selected_threshold, output):
    thresholds = np.asarray([row["threshold"] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for metric, label in (("precision_mean", "Precision"),
                          ("recall_mean", "Recall"),
                          ("f1_mean", "F1"), ("fpr_mean", "FPR")):
        ax.plot(thresholds, [row[metric] for row in rows], linewidth=2,
                label=label)
    ax.axvline(selected_threshold, color="#222222", linestyle="--",
               label=f"Elegido: {selected_threshold:.3f}")
    ax.set(title="Métricas out-of-fold · promedio de cinco semillas",
           xlabel="Umbral de TinyModel", ylabel="Métrica", ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _train_oof_seed(X, y, flagged, folds, selection, seed):
    """Predecir cada fila de desarrollo con el fold que no la entrenó."""
    development = np.concatenate(folds)
    predictions = {}
    batch = None if selection["batch_size"] == "full" else int(selection["batch_size"])
    for fold_number, validation_indices in enumerate(folds, start=1):
        training_indices = np.setdiff1d(development, validation_indices,
                                        assume_unique=True)
        standardizer = Standardizer.fit(X[training_indices], FRAUD_FEATURES)
        model = Perceptron(
            len(FRAUD_FEATURES), activation="logistic", beta=EXPERIMENT["beta"],
            init_scale=EXPERIMENT["init_scale"], seed=seed)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fit(
                model, standardizer.transform(X[training_indices]), y[training_indices],
                optimizer=make_optimizer(selection["optimizer"],
                                         selection["learning_rate"]),
                batch_size=batch, max_epochs=selection["max_epochs"],
                target_mse=0.0, shuffle=True, seed=seed)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fold_predictions = model.predict(
                standardizer.transform(X[validation_indices]))[:, 0]
        for index, probability in zip(validation_indices, fold_predictions):
            if int(index) in predictions:
                raise ValueError("una fila recibió más de una predicción out-of-fold")
            predictions[int(index)] = (float(probability), fold_number)
    expected = set(int(index) for index in development)
    if set(predictions) != expected:
        raise ValueError("las predicciones out-of-fold no cubren development")
    rows = [{
        "source_index": index,
        "seed": seed,
        "fold": predictions[index][1],
        "big_model_probability": float(y[index, 0]),
        "tiny_model_probability": predictions[index][0],
        "flagged_fraud": int(flagged[index]),
    } for index in sorted(expected)]
    return rows


def select_threshold(data_path, output):
    """Elegir y congelar umbral sólo con predicciones out-of-fold."""
    output = Path(output)
    if (output / "final-test.json").exists():
        raise ValueError("test ya fue evaluado; no se puede elegir el umbral después")
    threshold_path = output / "threshold-selection.json"
    if threshold_path.exists():
        raise ValueError("threshold-selection.json ya existe: el umbral ya está congelado")
    metadata = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if digest != metadata["data_sha256"]:
        raise ValueError("el dataset no coincide con el usado durante la selección")
    with np.load(output / "reserved-indices.npz", allow_pickle=False) as indices:
        development = indices["development"]
        folds = [indices[f"fold_{number}"]
                 for number in range(1, EXPERIMENT["folds"] + 1)]
    X, y, flagged = _load_fraud_file(data_path)
    selection = metadata["selection"]

    oof_rows = []
    probabilities = {}
    development_labels = flagged[development]
    positions = {int(index): position for position, index in enumerate(development)}
    for seed in EXPERIMENT["seeds"]:
        print(f"threshold: predicciones OOF semilla={seed}", flush=True)
        seed_rows = _train_oof_seed(X, y, flagged, folds, selection, seed)
        oof_rows.extend(seed_rows)
        ordered = np.empty(len(development), dtype=np.float64)
        for row in seed_rows:
            ordered[positions[row["source_index"]]] = row["tiny_model_probability"]
        probabilities[seed] = ordered

    coarse = np.round(np.arange(
        EXPERIMENT["coarse_threshold_start"],
        EXPERIMENT["coarse_threshold_stop"] + EXPERIMENT["coarse_threshold_step"] / 2,
        EXPERIMENT["coarse_threshold_step"]), 6)
    coarse_rows = evaluate_thresholds(
        development_labels, probabilities[0], coarse, seed=0)
    provisional = choose_threshold(coarse_rows)["threshold"]
    fine_start = max(0.001, provisional - EXPERIMENT["fine_threshold_radius"])
    fine_stop = min(0.999, provisional + EXPERIMENT["fine_threshold_radius"])
    fine = np.round(np.arange(
        fine_start, fine_stop + EXPERIMENT["fine_threshold_step"] / 2,
        EXPERIMENT["fine_threshold_step"]), 6)
    fine_rows = []
    per_seed = []
    for seed in EXPERIMENT["seeds"]:
        rows = evaluate_thresholds(
            development_labels, probabilities[seed], fine, seed=seed)
        fine_rows.extend(rows)
        best = choose_threshold(rows)
        per_seed.append({
            "seed": seed,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "fpr": best["fpr"],
        })
    summary_rows = _aggregate_thresholds(fine_rows)
    comparable = [{
        "threshold": row["threshold"],
        "f1": row["f1_mean"],
        "fpr": row["fpr_mean"],
        "recall": row["recall_mean"],
    } for row in summary_rows]
    selected = choose_threshold(comparable)
    selected_threshold = selected["threshold"]
    seed_zero_selected = next(
        row for row in fine_rows
        if row["seed"] == 0 and row["threshold"] == selected_threshold)
    mean_selected = next(
        row for row in summary_rows if row["threshold"] == selected_threshold)
    frozen = {
        "selection_source": "out-of-fold predictions from development only",
        "selection_metric": "mean F1 over five seeds",
        "coarse_seed_zero_optimum": provisional,
        "fine_search_start": float(fine_start),
        "fine_search_stop": float(fine_stop),
        "fine_search_step": EXPERIMENT["fine_threshold_step"],
        "threshold": selected_threshold,
        "metrics_mean_over_seeds": mean_selected,
        "seed_zero_confusion_matrix": [
            [seed_zero_selected["true_negative"], seed_zero_selected["false_positive"]],
            [seed_zero_selected["false_negative"], seed_zero_selected["true_positive"]],
        ],
        "seed_zero_metrics": seed_zero_selected,
        "per_seed_optima": per_seed,
        "test_was_evaluated_when_frozen": False,
    }
    write_csv(output / "oof-predictions.csv", oof_rows)
    write_csv(output / "threshold-coarse-seed-0.csv", coarse_rows)
    write_csv(output / "threshold-fine-by-seed.csv", fine_rows)
    write_csv(output / "threshold-fine-summary.csv", summary_rows)
    write_csv(output / "threshold-optimum-by-seed.csv", per_seed)
    threshold_path.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_thresholds(summary_rows, selected_threshold,
                     output / "threshold-validation.png")
    print(f"Umbral congelado sin test: {selected_threshold:.3f}", flush=True)
    return frozen


def search(data_path, output):
    """Elegir configuración usando desarrollo; no calcular nada sobre test."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output debe ser una carpeta nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    X, y, flagged = _load_fraud_file(data_path)
    development, test = _stratified_indices(
        flagged, EXPERIMENT["test_fraction"], EXPERIMENT["test_seed"])
    folds = stratified_folds(
        flagged, development, folds=EXPERIMENT["folds"],
        seed=EXPERIMENT["fold_seed"])
    np.savez_compressed(output / "reserved-indices.npz", development=development,
                        test=test, **{f"fold_{i + 1}": fold
                                      for i, fold in enumerate(folds)})

    summaries, curves, fold_details = [], [], []
    for optimizer, rates in EXPERIMENT["optimizer_learning_rates"].items():
        for rate in rates:
            _run_cv(
                X, y, folds, summaries, curves, fold_details,
                stage="optimizer_screen",
                optimizer_name=optimizer, learning_rate=rate,
                batch_size=EXPERIMENT["batch_size"], seed=0,
                max_epochs=EXPERIMENT["screen_epochs"])

    optimizer_screen = [row for row in summaries
                        if row["stage"] == "optimizer_screen"]
    best_per_optimizer = []
    for optimizer in EXPERIMENT["optimizer_learning_rates"]:
        optimizer_rows = [row for row in optimizer_screen
                          if row["optimizer"] == optimizer]
        if not optimizer_rows:
            raise ValueError(f"todas las tasas de {optimizer} divergieron")
        best_per_optimizer.append(_rank(optimizer_rows)[0])
    selected_optimizers = _rank(best_per_optimizer)[:EXPERIMENT["optimizer_finalists"]]

    for candidate in selected_optimizers:
        batch_value = candidate["batch_size"]
        for batch in EXPERIMENT["batch_sizes"]:
            if str(batch_value) == str("full" if batch is None else batch):
                continue
            _run_cv(
                X, y, folds, summaries, curves, fold_details,
                stage="batch_screen",
                optimizer_name=candidate["optimizer"],
                learning_rate=candidate["learning_rate"], batch_size=batch,
                seed=0,
                max_epochs=EXPERIMENT["batch_epochs"][
                    "full" if batch is None else str(batch)])

    batch_candidates = selected_optimizers + [
        row for row in summaries if row["stage"] == "batch_screen"]
    ranked_batches = _rank(batch_candidates)
    best_batch = ranked_batches[0]
    one_standard_error = best_batch["validation_std"] / np.sqrt(EXPERIMENT["folds"])
    equivalent_batches = [
        row for row in ranked_batches
        if row["validation_mse"] <= best_batch["validation_mse"] + one_standard_error
    ]
    equivalent_batches.sort(key=lambda row: (
        row["estimated_seconds_to_selected_epoch"], row["tail_range"],
        row["validation_mse"], row["configuration"],
    ))
    # Se conserva además el mínimo bruto aunque no esté entre los más rápidos,
    # para que el control final cuantifique el costo de esa diferencia diminuta.
    finalists = [best_batch]
    finalists.extend(row for row in equivalent_batches if row is not best_batch)
    finalists = finalists[:EXPERIMENT["configuration_finalists"]]
    for candidate in finalists:
        batch = None if candidate["batch_size"] == "full" else int(candidate["batch_size"])
        for seed in EXPERIMENT["seeds"]:
            if seed == 0:
                # La corrida seed 0 ya existe; se repite bajo la misma etapa para
                # que el agregado finalista sea explícito y autosuficiente.
                pass
            _run_cv(
                X, y, folds, summaries, curves, fold_details,
                stage="seed_finalist",
                optimizer_name=candidate["optimizer"],
                learning_rate=candidate["learning_rate"], batch_size=batch,
                seed=seed, max_epochs=candidate["epochs_run"])

    finalist_rows = [row for row in summaries if row["stage"] == "seed_finalist"]
    ranking = _aggregate_finalists(finalist_rows)
    best_validation = ranking[0]
    final_one_standard_error = (
        best_validation["validation_fold_std_mean"] / np.sqrt(EXPERIMENT["folds"]))
    equivalent_finalists = [
        row for row in ranking
        if row["validation_mse"] <= best_validation["validation_mse"]
        + final_one_standard_error
    ]
    equivalent_finalists.sort(key=lambda row: (
        row["estimated_seconds_to_selected_epoch"], row["tail_range_mean"],
        row["validation_seed_std"], row["validation_mse"],
        row["configuration"],
    ))
    winner = equivalent_finalists[0]
    selection = {
        "optimizer": winner["optimizer"],
        "learning_rate": winner["learning_rate"],
        "batch_size": winner["batch_size"],
        "max_epochs": max(1, winner["selected_epoch"]),
        "model_seed": 0,
        "shuffle_seed": 0,
        "selection_metric": "mean cross-validation MSE over five folds and five seeds",
        "validation_mse": winner["validation_mse"],
        "validation_seed_std": winner["validation_seed_std"],
        "validation_fold_std_mean": winner["validation_fold_std_mean"],
        "selection_rule": (
            "one-standard-error equivalence; then estimated time to stable epoch, "
            "tail stability, seed stability and mean validation MSE"
        ),
    }
    metadata = {
        "experiment": EXPERIMENT,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "development_samples": int(len(development)),
        "test_samples_reserved_and_not_evaluated": int(len(test)),
        "selection": selection,
    }
    write_csv(output / "cv-summaries.csv", summaries)
    write_csv(output / "cv-curves.csv", curves)
    write_csv(output / "cv-folds.csv", fold_details)
    write_csv(output / "finalist-ranking.csv", ranking)
    (output / "selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    _plot_curves(curves, [row["configuration"] for row in best_per_optimizer],
                 output / "optimizer-comparison.png",
                 "Mejor learning rate de cada optimizador",
                 stages={"optimizer_screen"})
    _plot_curves(curves, [row["configuration"] for row in batch_candidates],
                 output / "batch-comparison.png",
                 "Tamaños de batch para los optimizadores finalistas",
                 stages={"optimizer_screen", "batch_screen"})
    _plot_curves(curves, [row["configuration"] for row in finalists],
                 output / "finalists.png",
                 "Configuraciones finalistas · promedio de cinco semillas",
                 stages={"seed_finalist"})
    print(f"Ganadora sin abrir test: {winner['configuration']} · "
          f"{winner['selected_epoch']} épocas", flush=True)
    return selection


def finalize(data_path, output):
    """Reentrenar modelo y evaluar MSE/clasificación de test una sola vez."""
    output = Path(output)
    result_path = output / "final-test.json"
    if result_path.exists():
        raise ValueError("final-test.json ya existe: test ya fue evaluado")
    metadata = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    threshold_metadata = json.loads(
        (output / "threshold-selection.json").read_text(encoding="utf-8"))
    if threshold_metadata.get("test_was_evaluated_when_frozen") is not False:
        raise ValueError("el umbral debe quedar congelado antes de evaluar test")
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if digest != metadata["data_sha256"]:
        raise ValueError("el dataset no coincide con el usado durante la selección")
    with np.load(output / "reserved-indices.npz", allow_pickle=False) as indices:
        development = indices["development"]
        test = indices["test"]
    X, y, flagged = _load_fraud_file(data_path)
    standardizer = Standardizer.fit(X[development], FRAUD_FEATURES)
    X_development = standardizer.transform(X[development])
    X_test = standardizer.transform(X[test])
    selection = metadata["selection"]
    batch = None if selection["batch_size"] == "full" else int(selection["batch_size"])
    model = Perceptron(
        len(FRAUD_FEATURES), activation="logistic", beta=EXPERIMENT["beta"],
        init_scale=EXPERIMENT["init_scale"], seed=selection["model_seed"])
    started = perf_counter()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        history = fit(
            model, X_development, y[development],
            optimizer=make_optimizer(selection["optimizer"], selection["learning_rate"]),
            batch_size=batch, max_epochs=selection["max_epochs"], target_mse=0.0,
            shuffle=True, seed=selection["shuffle_seed"])
    seconds = perf_counter() - started
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        development_prediction = model.predict(X_development)
        test_prediction = model.predict(X_test)
    threshold = float(threshold_metadata["threshold"])
    predicted_test = (test_prediction[:, 0] >= threshold).astype(np.int64)
    test_report = classification_metrics(
        flagged[test], predicted_test, labels=(0, 1))
    test_positive = test_report.for_label(1)
    big_model_predicted = (y[test, 0] >= 0.85).astype(np.int64)
    big_model_report = classification_metrics(
        flagged[test], big_model_predicted, labels=(0, 1))
    big_model_positive = big_model_report.for_label(1)
    result = {
        "configuration": selection,
        "threshold": threshold,
        "development_samples": int(len(development)),
        "test_samples": int(len(test)),
        "development_mse": float(np.mean((development_prediction - y[development]) ** 2)),
        "test_mse": float(np.mean((test_prediction - y[test]) ** 2)),
        "classification": {
            "confusion_matrix": test_report.confusion_matrix.tolist(),
            "accuracy": test_report.accuracy,
            "precision": test_positive.precision,
            "recall": test_positive.recall,
            "f1": test_positive.f1,
            "tpr": test_positive.tpr,
            "fpr": test_positive.fpr,
        },
        "big_model_at_0_85": {
            "confusion_matrix": big_model_report.confusion_matrix.tolist(),
            "accuracy": big_model_report.accuracy,
            "precision": big_model_positive.precision,
            "recall": big_model_positive.recall,
            "f1": big_model_positive.f1,
            "tpr": big_model_positive.tpr,
            "fpr": big_model_positive.fpr,
        },
        "training_seconds": seconds,
        "test_evaluations": 1,
    }
    standardizer.save(output / "final-standardizer.npz")
    model.save(output / "final-model.npz")
    write_csv(output / "final-history.csv", history)
    write_csv(output / "final-test-predictions.csv", [{
        "source_index": int(index),
        "big_model_probability": float(expected[0]),
        "tiny_model_probability": float(predicted[0]),
        "flagged_fraud": int(label),
        "predicted_fraud": int(predicted_label),
    } for index, expected, predicted, label, predicted_label in zip(
        test, y[test], test_prediction, flagged[test], predicted_test)])
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Test evaluado una vez: MSE={result['test_mse']:.8f} · "
          f"F1={test_positive.f1:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fraud_dataset.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("search", "threshold", "finalize"),
                        required=True)
    args = parser.parse_args()
    try:
        if args.stage == "search":
            search(args.data, args.output)
        elif args.stage == "threshold":
            select_threshold(args.data, args.output)
        else:
            finalize(args.data, args.output)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

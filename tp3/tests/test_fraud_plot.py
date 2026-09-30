"""Probar la lectura y generación de curvas del experimento de fraude."""

import csv
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="los gráficos requieren el extra plot")


spec = importlib.util.spec_from_file_location(
    "plot_fraud_experiment",
    Path(__file__).resolve().parents[1] / "scripts/plot_fraud_experiment.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def write_history(path, validation=(0.4, 0.2, 0.25)):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "mse", "validation_mse"])
        writer.writeheader()
        for epoch, value in enumerate(validation):
            training = 0.4 / (epoch + 1) if value == "" else value / 2
            writer.writerow({"epoch": epoch, "mse": training,
                             "validation_mse": value})


def test_load_history_and_plot_run(tmp_path):
    run = tmp_path / "run"
    model = run / "logistic"
    model.mkdir(parents=True)
    (run / "summary.json").write_text('[{"model": "logistic"}]')
    write_history(model / "history.csv")

    output = tmp_path / "plots"
    diagnostics = plots.plot_run(run, output)

    assert diagnostics[0]["best_validation_epoch"] == 1
    assert diagnostics[0]["best_validation_mse"] == pytest.approx(0.2)
    assert (output / "learning-curves.png").is_file()
    assert (output / "learning-curves-log.png").is_file()
    assert (output / "learning-diagnostics.csv").is_file()


def test_plot_run_accepts_learning_without_validation(tmp_path):
    run = tmp_path / "run"
    model = run / "linear"
    model.mkdir(parents=True)
    (run / "summary.json").write_text('[{"model": "linear"}]')
    write_history(model / "history.csv", validation=("", "", ""))

    diagnostics = plots.plot_run(run, tmp_path / "plots")

    assert diagnostics[0]["best_training_epoch"] == 2
    assert diagnostics[0]["best_validation_epoch"] is None


def test_load_history_rejects_nonconsecutive_epochs(tmp_path):
    path = tmp_path / "history.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "mse", "validation_mse"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "mse": 1, "validation_mse": 1})
        writer.writerow({"epoch": 2, "mse": 0.5, "validation_mse": 0.5})
    with pytest.raises(ValueError, match="consecutivas"):
        plots.load_history(path)

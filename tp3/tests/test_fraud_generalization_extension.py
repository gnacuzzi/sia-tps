"""Pruebas del protocolo extendido de generalización."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("matplotlib", reason="el análisis genera gráficos")
spec = importlib.util.spec_from_file_location(
    "analyze_fraud_generalization_extension",
    Path(__file__).resolve().parents[1] / "scripts/analyze_fraud_generalization_extension.py",
)
generalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generalization)


def test_stratified_folds_are_exhaustive_disjoint_and_balanced():
    labels = np.asarray([0] * 15 + [1] * 10)
    indices = np.arange(len(labels))
    folds = generalization.stratified_folds(labels, indices, folds=5, seed=3)
    assert sorted(np.concatenate(folds).tolist()) == indices.tolist()
    assert sum(len(fold) for fold in folds) == len(set(np.concatenate(folds)))
    assert [int(labels[fold].sum()) for fold in folds] == [2] * 5


def test_stable_epoch_requires_the_curve_to_remain_inside_band():
    assert generalization.stable_epoch([2.0, 1.0005, 1.1, 1.0, 1.0002]) == 3


def test_threshold_metrics_and_selection_use_binary_predictions():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.6, 0.7, 0.9])
    rows = generalization.evaluate_thresholds(
        labels, probabilities, [0.5, 0.65, 0.8], seed=0)
    chosen = generalization.choose_threshold(rows)
    assert chosen["threshold"] == pytest.approx(0.65)
    assert chosen["true_negative"] == 2
    assert chosen["true_positive"] == 2
    assert chosen["false_positive"] == 0
    assert chosen["false_negative"] == 0


def test_threshold_selection_rejects_unknown_metric():
    with pytest.raises(ValueError, match="métrica"):
        generalization.choose_threshold([{"f1": 1.0}], metric="mse")


@pytest.mark.parametrize("name", [
    "gradient_descent", "momentum", "adaptive", "rmsprop", "adam",
])
def test_all_declared_optimizers_can_be_built(name):
    assert generalization.make_optimizer(name, 0.001).learning_rate == 0.001

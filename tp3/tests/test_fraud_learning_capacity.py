"""Pruebas del estudio extendido de capacidad del ejercicio 1."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("matplotlib", reason="el análisis genera gráficos")
spec = importlib.util.spec_from_file_location(
    "analyze_fraud_learning_capacity",
    Path(__file__).resolve().parents[1] / "scripts/analyze_fraud_learning_capacity.py",
)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def test_plateau_requires_remaining_inside_band():
    # La época 1 entra momentáneamente en la banda, pero vuelve a salir.
    assert capacity.plateau_epoch([10, 1.005, 1.2, 1.0, 1.004]) == 3


def test_plateau_rejects_invalid_series():
    with pytest.raises(ValueError, match="serie finita"):
        capacity.plateau_epoch([1, np.nan])


@pytest.mark.parametrize("name", [
    "gradient_descent", "momentum", "adaptive", "rmsprop", "adam",
])
def test_every_optimizer_declared_can_be_built(name):
    optimizer = capacity.make_optimizer(name, 0.001)
    assert optimizer.learning_rate == pytest.approx(0.001)

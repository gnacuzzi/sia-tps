"""Regla de Oja para aproximar la primera componente principal."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence, Tuple

import numpy as np

from .pca import Standardizer


@dataclass(frozen=True)
class OjaConfig:
    """Hiperparámetros del aprendizaje en línea de Oja."""

    epochs: int
    learning_rate_start: float
    learning_rate_end: float
    shuffle: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, Integral)
            or self.epochs < 1
        ):
            raise ValueError("epochs debe ser un entero positivo")
        if not (
            0 < self.learning_rate_end <= self.learning_rate_start <= 0.5
        ):
            raise ValueError(
                "las tasas deben cumplir 0 < final <= inicial <= 0.5"
            )
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle debe ser booleano")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed debe ser un entero")


@dataclass(frozen=True)
class OjaModel:
    """Modelo entrenado y trayectoria de pesos al final de cada época."""

    weights: np.ndarray
    weight_history: np.ndarray
    feature_names: Tuple[str, ...]
    standardizer: Standardizer
    config: OjaConfig

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Proyectar filas en unidades originales sobre la dirección aprendida."""

        standardized = self.standardizer.transform(values)
        # Accelerate puede emitir advertencias de punto flotante espurias en
        # ``matmul``. Como en PCA, validamos el resultado en lugar de ocultar
        # una proyección realmente inválida.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            projections = standardized @ self.weights
        if not np.isfinite(projections).all():
            raise FloatingPointError("la proyección de Oja produjo valores no finitos")
        return projections

    @property
    def weight_norm_history(self) -> np.ndarray:
        """Norma de los pesos iniciales y al finalizar cada época."""

        return np.linalg.norm(self.weight_history, axis=1)


def _linear_schedule(start: float, end: float, step: int, steps: int) -> float:
    if steps == 1:
        return start
    progress = step / (steps - 1)
    return start + progress * (end - start)


def _oja_update(weights: np.ndarray, sample: np.ndarray, rate: float) -> np.ndarray:
    """Aplicar una actualización en línea de la regla vista en clase."""

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        output = float(sample @ weights)
        return weights + rate * output * (sample - output * weights)


def align_sign(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Orientar un vector hacia una referencia sin alterar su dirección."""

    vector = np.asarray(vector, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if vector.ndim != 1 or vector.shape != reference.shape:
        raise ValueError("los vectores deben ser unidimensionales y del mismo tamaño")
    if not np.isfinite(vector).all() or not np.isfinite(reference).all():
        raise ValueError("los vectores deben contener valores finitos")
    if np.linalg.norm(vector) == 0 or np.linalg.norm(reference) == 0:
        raise ValueError("no se puede alinear un vector nulo")
    return vector.copy() if vector @ reference >= 0 else -vector


def direction_similarity(vector: np.ndarray, reference: np.ndarray) -> float:
    """Calcular el coseno absoluto, ignorando la ambigüedad de signo."""

    aligned = align_sign(vector, reference)
    denominator = np.linalg.norm(aligned) * np.linalg.norm(reference)
    return float((aligned @ reference) / denominator)


def fit_oja(
    values: np.ndarray,
    feature_names: Sequence[str],
    config: OjaConfig,
) -> OjaModel:
    """Ajustar Oja sobre los z-scores de las entradas.

    Los pesos se inicializan con una distribución uniforme entre 0 y 1, tal
    como se propone en clase. La tasa cambia linealmente por presentación de
    muestra; si sus extremos coinciden, permanece constante. No se normalizan
    los pesos de forma explícita: el término correctivo de Oja debe mantenerlos
    acotados y llevar su norma hacia uno.
    """

    standardizer = Standardizer.fit(values, feature_names)
    standardized = standardizer.transform(values)
    generator = np.random.default_rng(config.seed)
    weights = generator.uniform(0.0, 1.0, size=standardized.shape[1])
    history = [weights.copy()]
    total_steps = config.epochs * standardized.shape[0]
    step = 0

    for epoch in range(config.epochs):
        order = (
            generator.permutation(standardized.shape[0])
            if config.shuffle
            else np.arange(standardized.shape[0])
        )
        for sample_index in order:
            rate = _linear_schedule(
                config.learning_rate_start,
                config.learning_rate_end,
                step,
                total_steps,
            )
            weights = _oja_update(weights, standardized[sample_index], rate)
            if not np.isfinite(weights).all():
                raise FloatingPointError(
                    "Oja divergió y produjo pesos no finitos "
                    f"en la época {epoch + 1}"
                )
            step += 1
        history.append(weights.copy())

    return OjaModel(
        weights=weights,
        weight_history=np.asarray(history),
        feature_names=tuple(feature_names),
        standardizer=standardizer,
        config=config,
    )

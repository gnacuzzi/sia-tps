"""PCA mediante autodescomposición de la covarianza de datos estandarizados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    """Parámetros ajustados por columna para reutilizar con datos futuros."""

    feature_names: Tuple[str, ...]
    means: np.ndarray
    standard_deviations: np.ndarray

    @classmethod
    def fit(
        cls, values: np.ndarray, feature_names: Sequence[str]
    ) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        names = tuple(feature_names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("la matriz y los nombres de variables no coinciden")
        if values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("se necesitan al menos dos filas con valores finitos")
        means = values.mean(axis=0)
        deviations = values.std(axis=0, ddof=1)
        if np.any(deviations == 0):
            raise ValueError("no se pueden estandarizar variables constantes")
        return cls(names, means, deviations)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("la matriz no tiene las variables esperadas")
        if not np.isfinite(values).all():
            raise ValueError("la matriz contiene valores no finitos")
        return (values - self.means) / self.standard_deviations

    def inverse_transform(self, standardized: np.ndarray) -> np.ndarray:
        standardized = np.asarray(standardized, dtype=np.float64)
        if standardized.ndim != 2 or standardized.shape[1] != len(
            self.feature_names
        ):
            raise ValueError("la matriz no tiene las variables esperadas")
        return standardized * self.standard_deviations + self.means


@dataclass(frozen=True)
class PCAResult:
    """Resultados completos; las componentes están ordenadas por varianza."""

    feature_names: Tuple[str, ...]
    standardized_values: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    scores: np.ndarray
    loadings: np.ndarray
    standardizer: Standardizer

    def component_count(self, minimum_cumulative_variance: float) -> int:
        if not 0 < minimum_cumulative_variance <= 1:
            raise ValueError("el umbral de varianza debe pertenecer a (0, 1]")
        count = int(
            np.searchsorted(
                self.cumulative_explained_variance,
                minimum_cumulative_variance,
                side="left",
            )
            + 1
        )
        return min(count, len(self.cumulative_explained_variance))


def _orient_components(components: np.ndarray) -> np.ndarray:
    """Fijar el signo haciendo positiva la carga de mayor magnitud."""

    oriented = components.copy()
    for index in range(oriented.shape[1]):
        pivot = int(np.argmax(np.abs(oriented[:, index])))
        if oriented[pivot, index] < 0:
            oriented[:, index] *= -1
    return oriented


def fit_pca(values: np.ndarray, feature_names: Sequence[str]) -> PCAResult:
    """Ajustar PCA sobre z-scores usando la covarianza muestral y ``eigh``."""

    standardizer = Standardizer.fit(values, feature_names)
    standardized = standardizer.transform(values)
    covariance = np.cov(standardized, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = _orient_components(eigenvectors[:, order])
    total_variance = eigenvalues.sum()
    if total_variance <= 0:
        raise ValueError("la matriz no contiene varianza para aplicar PCA")
    explained = eigenvalues / total_variance
    cumulative = np.cumsum(explained)
    cumulative[-1] = 1.0
    # Accelerate puede emitir advertencias de punto flotante espurias en
    # ``matmul``. Validamos explícitamente el resultado en lugar de ocultar un
    # valor realmente inválido.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = standardized @ components
    if not np.isfinite(scores).all():
        raise FloatingPointError("la proyección PCA produjo scores no finitos")
    loadings = components * np.sqrt(eigenvalues)

    return PCAResult(
        feature_names=tuple(feature_names),
        standardized_values=standardized,
        covariance=covariance,
        eigenvalues=eigenvalues,
        components=components,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        scores=scores,
        loadings=loadings,
        standardizer=standardizer,
    )

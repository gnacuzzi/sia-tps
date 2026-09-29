"""Preprocesamiento ajustado únicamente con datos de training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    """Estandarizar columnas con media y desvío aprendidos de training.

    Las columnas constantes reciben escala 1: quedan centradas en cero sin
    provocar divisiones por cero. ``transform`` nunca vuelve a estimar los
    parámetros, por lo que puede aplicarse a test sin filtrar su distribución.
    """

    mean: np.ndarray
    scale: np.ndarray
    feature_names: Tuple[str, ...]

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        scale = np.asarray(self.scale, dtype=np.float64).copy()
        names = tuple(self.feature_names)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("mean y scale deben ser vectores de igual forma")
        if len(names) != len(mean) or len(set(names)) != len(names):
            raise ValueError("feature_names debe identificar cada columna una sola vez")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("mean y scale deben ser finitos y scale positivo")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_names", names)

    @classmethod
    def fit(cls, X, feature_names: Sequence[str]):
        """Calcular un parámetro por columna usando exclusivamente ``X``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X debe ser una matriz no vacía de valores finitos")
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        return cls(mean=mean, scale=np.where(std == 0, 1.0, std),
                   feature_names=tuple(feature_names))

    def transform(self, X):
        """Aplicar los parámetros guardados sin recalcularlos."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean):
            raise ValueError("X debe tener una columna por feature guardada")
        if not np.isfinite(X).all():
            raise ValueError("X debe contener solamente valores finitos")
        return (X - self.mean) / self.scale

    def save(self, path):
        """Guardar los parámetros necesarios para reproducir predicciones."""
        with Path(path).open("wb") as file:
            np.savez_compressed(file, mean=self.mean, scale=self.scale,
                                feature_names=np.asarray(self.feature_names))

    @classmethod
    def load(cls, path):
        """Recuperar un estandarizador sin pickle."""
        with np.load(path, allow_pickle=False) as data:
            required = {"mean", "scale", "feature_names"}
            if set(data.files) != required:
                raise ValueError("archivo de estandarización inválido")
            return cls(mean=data["mean"], scale=data["scale"],
                       feature_names=tuple(str(name) for name in data["feature_names"]))

"""Red de Kohonen rectangular con aprendizaje competitivo en línea."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Sequence, Tuple

import numpy as np

from .pca import Standardizer


Initialization = Literal["sample", "uniform"]


@dataclass(frozen=True)
class KohonenConfig:
    """Hiperparámetros que se compararán en los experimentos del TP."""

    grid_size: int
    epochs: int
    learning_rate_start: float
    learning_rate_end: float
    radius_start: float
    radius_end: float = 1.0
    initialization: Initialization = "sample"
    shuffle: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.grid_size, bool)
            or not isinstance(self.grid_size, Integral)
            or self.grid_size < 2
        ):
            raise ValueError("grid_size debe ser un entero mayor o igual que 2")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, Integral)
            or self.epochs < 1
        ):
            raise ValueError("epochs debe ser un entero positivo")
        if not 0 < self.learning_rate_end <= self.learning_rate_start < 1:
            raise ValueError(
                "las tasas deben cumplir 0 < final <= inicial < 1"
            )
        if not 1 <= self.radius_end <= self.radius_start:
            raise ValueError(
                "los radios deben cumplir 1 <= final <= inicial"
            )
        if self.initialization not in ("sample", "uniform"):
            raise ValueError("initialization debe ser 'sample' o 'uniform'")
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle debe ser booleano")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed debe ser un entero")


@dataclass(frozen=True)
class KohonenModel:
    """Mapa entrenado; los pesos se almacenan en el espacio estandarizado."""

    weights: np.ndarray
    feature_names: Tuple[str, ...]
    standardizer: Standardizer
    config: KohonenConfig

    def _standardized_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        squared_distances = np.sum((self.weights - sample) ** 2, axis=2)
        flat_index = int(np.argmin(squared_distances))
        row, column = np.unravel_index(flat_index, squared_distances.shape)
        return int(row), int(column)

    def bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Obtener la neurona ganadora de una muestra en unidades originales."""

        sample = np.asarray(sample, dtype=np.float64)
        if sample.ndim != 1 or sample.shape[0] != len(self.feature_names):
            raise ValueError("la muestra no tiene las variables esperadas")
        standardized = self.standardizer.transform(sample.reshape(1, -1))[0]
        return self._standardized_bmu(standardized)

    def map_samples(self, values: np.ndarray) -> np.ndarray:
        """Asignar cada fila, expresada en unidades originales, a su BMU."""

        standardized = self.standardizer.transform(values)
        return np.asarray(
            [self._standardized_bmu(sample) for sample in standardized],
            dtype=np.int64,
        )

    def hit_map(self, values: np.ndarray) -> np.ndarray:
        """Contar cuántas muestras se asignan a cada neurona."""

        hits = np.zeros((self.config.grid_size, self.config.grid_size), dtype=int)
        for row, column in self.map_samples(values):
            hits[row, column] += 1
        return hits

    def quantization_error(self, values: np.ndarray) -> float:
        """Promediar la distancia de cada muestra a su BMU.

        El cálculo se realiza en el mismo espacio estandarizado usado durante
        el entrenamiento, para que ninguna variable domine por su escala.
        """

        standardized = self.standardizer.transform(values)
        distances = []
        for sample in standardized:
            row, column = self._standardized_bmu(sample)
            distances.append(
                float(np.linalg.norm(sample - self.weights[row, column]))
            )
        return float(np.mean(distances))

    def u_matrix(self) -> np.ndarray:
        """Promediar distancias a las vecinas ortogonales de cada neurona."""

        size = self.config.grid_size
        matrix = np.zeros((size, size), dtype=np.float64)
        for row in range(size):
            for column in range(size):
                distances = []
                for row_delta, column_delta in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta
                    if 0 <= neighbor_row < size and 0 <= neighbor_column < size:
                        difference = (
                            self.weights[row, column]
                            - self.weights[neighbor_row, neighbor_column]
                        )
                        distances.append(float(np.linalg.norm(difference)))
                matrix[row, column] = float(np.mean(distances))
        return matrix

    def prototypes(self, original_scale: bool = True) -> np.ndarray:
        """Devolver los perfiles neuronales en escala original o estandarizada."""

        if not original_scale:
            return self.weights.copy()
        size = self.config.grid_size
        flat_weights = self.weights.reshape(size * size, -1)
        original = self.standardizer.inverse_transform(flat_weights)
        return original.reshape(self.weights.shape)

    def component_plane(
        self, feature_name: str, original_scale: bool = True
    ) -> np.ndarray:
        """Obtener el mapa de una variable para interpretar regiones."""

        try:
            feature_index = self.feature_names.index(feature_name)
        except ValueError as error:
            raise ValueError(f"variable desconocida: {feature_name}") from error
        return self.prototypes(original_scale=original_scale)[:, :, feature_index]


def _linear_schedule(start: float, end: float, step: int, steps: int) -> float:
    if steps == 1:
        return start
    progress = step / (steps - 1)
    return start + progress * (end - start)


def _neighborhood_mask(
    coordinates: np.ndarray, bmu: np.ndarray, radius: float
) -> np.ndarray:
    """Incluir las neuronas cuya distancia está en el borde del radio."""

    return np.linalg.norm(coordinates - bmu, axis=2) <= radius


def _initialize_weights(
    standardized: np.ndarray,
    config: KohonenConfig,
    generator: np.random.Generator,
) -> np.ndarray:
    neuron_count = config.grid_size**2
    if config.initialization == "sample":
        indices = generator.choice(
            standardized.shape[0],
            size=neuron_count,
            replace=neuron_count > standardized.shape[0],
        )
        return standardized[indices].reshape(
            config.grid_size, config.grid_size, standardized.shape[1]
        ).copy()

    minima = standardized.min(axis=0)
    maxima = standardized.max(axis=0)
    return generator.uniform(
        minima,
        maxima,
        size=(config.grid_size, config.grid_size, standardized.shape[1]),
    )


def fit_kohonen(
    values: np.ndarray,
    feature_names: Sequence[str],
    config: KohonenConfig,
) -> KohonenModel:
    """Ajustar una SOM ``k x k`` sobre los z-scores de las entradas.

    La tasa y el radio decrecen linealmente por presentación de muestra. La
    vecindad es rectangular y usa distancia euclídea entre coordenadas. Se
    incluye el borde del radio, de modo que ``R = 1`` contiene a la ganadora y
    a sus cuatro vecinas ortogonales cuando existen.
    """

    standardizer = Standardizer.fit(values, feature_names)
    standardized = standardizer.transform(values)
    generator = np.random.default_rng(config.seed)
    weights = _initialize_weights(standardized, config, generator)

    coordinates = np.indices((config.grid_size, config.grid_size)).transpose(
        1, 2, 0
    )
    total_steps = config.epochs * standardized.shape[0]
    step = 0

    for _ in range(config.epochs):
        order = (
            generator.permutation(standardized.shape[0])
            if config.shuffle
            else np.arange(standardized.shape[0])
        )
        for sample_index in order:
            sample = standardized[sample_index]
            squared_distances = np.sum((weights - sample) ** 2, axis=2)
            flat_bmu = int(np.argmin(squared_distances))
            bmu = np.asarray(np.unravel_index(flat_bmu, squared_distances.shape))

            learning_rate = _linear_schedule(
                config.learning_rate_start,
                config.learning_rate_end,
                step,
                total_steps,
            )
            radius = _linear_schedule(
                config.radius_start,
                config.radius_end,
                step,
                total_steps,
            )
            neighborhood = _neighborhood_mask(coordinates, bmu, radius)
            weights[neighborhood] += learning_rate * (
                sample - weights[neighborhood]
            )
            step += 1

    return KohonenModel(
        weights=weights,
        feature_names=tuple(feature_names),
        standardizer=standardizer,
        config=config,
    )

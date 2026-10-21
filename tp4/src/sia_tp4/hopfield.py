"""Red de Hopfield discreta para memorias asociativas bipolares."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Optional

import numpy as np


RecallStatus = Literal["fixed_point", "cycle", "max_iterations"]


def _as_bipolar_matrix(patterns: np.ndarray) -> np.ndarray:
    """Validar y copiar una colección de patrones almacenados por fila."""

    values = np.asarray(patterns)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(
            "patterns debe ser una matriz no vacía con un patrón por fila"
        )
    if values.dtype.kind not in "iuf":
        raise ValueError("patterns debe ser una matriz numérica")
    if not np.isfinite(values).all():
        raise ValueError("patterns debe contener valores finitos")
    if not np.isin(values, (-1, 1)).all():
        raise ValueError("patterns debe contener únicamente -1 y 1")
    return values.astype(np.int8, copy=True)


def _as_bipolar_state(state: np.ndarray, neuron_count: int) -> np.ndarray:
    """Validar y copiar un estado compatible con la red."""

    values = np.asarray(state)
    if values.ndim != 1 or values.shape[0] != neuron_count:
        raise ValueError(
            f"state debe ser un vector de longitud {neuron_count}"
        )
    if values.dtype.kind not in "iuf":
        raise ValueError("state debe ser un vector numérico")
    if not np.isfinite(values).all():
        raise ValueError("state debe contener valores finitos")
    if not np.isin(values, (-1, 1)).all():
        raise ValueError("state debe contener únicamente -1 y 1")
    return values.astype(np.int8, copy=True)


def _update_state(weights: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Aplicar sincrónicamente la regla de activación vista en clase.

    Cuando el campo local de una neurona es cero, se conserva su estado
    anterior en lugar de asignarle arbitrariamente uno de los dos signos.
    """

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        local_fields = weights @ state
    if not np.isfinite(local_fields).all():
        raise FloatingPointError(
            "la actualización de Hopfield produjo activaciones no finitas"
        )
    return np.where(
        local_fields > 0,
        1,
        np.where(local_fields < 0, -1, state),
    ).astype(np.int8)


@dataclass(frozen=True)
class RecallResult:
    """Trayectoria completa de una consulta a la memoria asociativa."""

    final_state: np.ndarray
    states: np.ndarray
    energies: np.ndarray
    iterations: int
    status: RecallStatus
    cycle_length: Optional[int] = None

    @property
    def converged(self) -> bool:
        """Indicar si la trayectoria terminó en un estado fijo."""

        return self.status == "fixed_point"


@dataclass(frozen=True)
class HopfieldModel:
    """Pesos fijos y patrones almacenados de una red de Hopfield discreta."""

    weights: np.ndarray
    stored_patterns: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights)
        if (
            weights.ndim != 2
            or weights.shape[0] < 1
            or weights.shape[0] != weights.shape[1]
        ):
            raise ValueError("weights debe ser una matriz cuadrada no vacía")
        if weights.dtype.kind not in "iuf" or not np.isfinite(weights).all():
            raise ValueError("weights debe contener valores numéricos finitos")
        weights = weights.astype(np.float64, copy=True)
        if not np.allclose(weights, weights.T):
            raise ValueError("weights debe ser simétrica")
        if not np.allclose(np.diag(weights), 0.0):
            raise ValueError("la diagonal de weights debe ser cero")

        stored_patterns = _as_bipolar_matrix(self.stored_patterns)
        if stored_patterns.shape[1] != weights.shape[0]:
            raise ValueError(
                "stored_patterns y weights deben tener la misma cantidad "
                "de neuronas"
            )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "stored_patterns", stored_patterns)

    @property
    def neuron_count(self) -> int:
        """Cantidad de neuronas, igual a la longitud de cada patrón."""

        return int(self.weights.shape[0])

    def energy(self, state: np.ndarray) -> float:
        """Calcular ``-1/2 * sum_ij(w_ij * s_i * s_j)``."""

        checked = _as_bipolar_state(state, self.neuron_count)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            value = -0.5 * np.einsum(
                "i,ij,j->", checked, self.weights, checked
            )
        if not np.isfinite(value):
            raise FloatingPointError(
                "el cálculo de energía produjo un valor no finito"
            )
        return float(value)

    def matching_pattern_index(self, state: np.ndarray) -> Optional[int]:
        """Devolver el índice del patrón almacenado que coincide exactamente."""

        checked = _as_bipolar_state(state, self.neuron_count)
        matches = np.flatnonzero(np.all(self.stored_patterns == checked, axis=1))
        return int(matches[0]) if matches.size else None

    def recall(
        self, query: np.ndarray, max_iterations: int = 100
    ) -> RecallResult:
        """Actualizar una consulta hasta un estado fijo, un ciclo o el límite.

        ``states`` y ``energies`` incluyen el estado inicial y el estado
        obtenido después de cada actualización. Por eso contienen
        ``iterations + 1`` elementos, incluso cuando el último estado repite
        al anterior o cierra un ciclo.
        """

        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, Integral)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations debe ser un entero positivo")

        state = _as_bipolar_state(query, self.neuron_count)
        states = [state.copy()]
        energies = [self.energy(state)]
        seen_at = {state.tobytes(): 0}

        for iteration in range(1, int(max_iterations) + 1):
            next_state = _update_state(self.weights, state)
            states.append(next_state.copy())
            energies.append(self.energy(next_state))

            if np.array_equal(next_state, state):
                return RecallResult(
                    final_state=next_state.copy(),
                    states=np.asarray(states, dtype=np.int8),
                    energies=np.asarray(energies, dtype=np.float64),
                    iterations=iteration,
                    status="fixed_point",
                )

            state_key = next_state.tobytes()
            if state_key in seen_at:
                return RecallResult(
                    final_state=next_state.copy(),
                    states=np.asarray(states, dtype=np.int8),
                    energies=np.asarray(energies, dtype=np.float64),
                    iterations=iteration,
                    status="cycle",
                    cycle_length=iteration - seen_at[state_key],
                )

            seen_at[state_key] = iteration
            state = next_state

        return RecallResult(
            final_state=state.copy(),
            states=np.asarray(states, dtype=np.int8),
            energies=np.asarray(energies, dtype=np.float64),
            iterations=int(max_iterations),
            status="max_iterations",
        )


def fit_hopfield(patterns: np.ndarray) -> HopfieldModel:
    """Almacenar patrones bipolares mediante la regla de Hebb de la clase."""

    stored_patterns = _as_bipolar_matrix(patterns)
    neuron_count = stored_patterns.shape[1]
    float_patterns = stored_patterns.astype(np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        weights = (float_patterns.T @ float_patterns) / neuron_count
    np.fill_diagonal(weights, 0.0)
    if not np.isfinite(weights).all():
        raise FloatingPointError(
            "el cálculo de pesos de Hopfield produjo valores no finitos"
        )
    return HopfieldModel(
        weights=weights,
        stored_patterns=stored_patterns,
    )

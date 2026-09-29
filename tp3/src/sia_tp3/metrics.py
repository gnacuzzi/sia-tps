"""Métricas estándar de clasificación derivadas de una matriz de confusión."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ClassMetrics:
    """Conteos y métricas de una clase considerada positiva contra el resto."""

    label: int
    support: int
    predicted: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    tpr: float
    fpr: float


@dataclass(frozen=True)
class ClassificationReport:
    """Resultado global y por clase; filas reales, columnas predichas."""

    labels: Tuple[int, ...]
    confusion_matrix: np.ndarray
    accuracy: float
    per_class: Tuple[ClassMetrics, ...]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_tpr: float
    macro_fpr: float

    def for_label(self, label: int) -> ClassMetrics:
        """Obtener las métricas de una clase explícita."""
        for metrics in self.per_class:
            if metrics.label == label:
                return metrics
        raise KeyError(f"clase desconocida: {label}")


def _validated_labels(y_true, y_pred, labels: Sequence[int]):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true e y_pred deben ser vectores de clases")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener igual longitud no vacía")
    for values in (y_true, y_pred):
        if not (np.issubdtype(values.dtype, np.integer) or
                np.issubdtype(values.dtype, np.bool_)):
            raise ValueError("las clases reales y predichas deben ser enteras")

    labels = tuple(int(label) for label in labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels debe contener clases únicas")
    known = set(labels)
    observed = set(int(value) for value in np.concatenate([y_true, y_pred]))
    unknown = observed - known
    if unknown:
        raise ValueError(f"hay clases que no figuran en labels: {sorted(unknown)}")
    return y_true.astype(np.int64), y_pred.astype(np.int64), labels


def _confusion_matrix_validated(y_true, y_pred, labels):
    indices = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    rows = np.fromiter((indices[int(value)] for value in y_true), dtype=np.int64)
    columns = np.fromiter((indices[int(value)] for value in y_pred), dtype=np.int64)
    np.add.at(matrix, (rows, columns), 1)
    return matrix


def confusion_matrix(y_true, y_pred, *, labels: Sequence[int]) -> np.ndarray:
    """Contar casos con filas reales y columnas predichas.

    ``labels`` es obligatorio para conservar clases ausentes, como el dígito 8
    en training, en vez de ocultarlas al inferir sólo las clases observadas.
    """
    y_true, y_pred, labels = _validated_labels(y_true, y_pred, labels)
    return _confusion_matrix_validated(y_true, y_pred, labels)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _macro(values) -> float:
    values = np.asarray(tuple(values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def classification_metrics(y_true, y_pred, *,
                           labels: Sequence[int]) -> ClassificationReport:
    """Calcular accuracy y métricas one-vs-rest por clase.

    Las entradas ya deben ser clases. La conversión desde probabilidades por
    umbral o desde salidas multiclase por ``argmax`` queda fuera de esta función.
    Una métrica con denominador cero se informa como ``NaN`` porque no puede
    evaluarse con las muestras recibidas. TPR coincide deliberadamente con recall.
    """
    y_true, y_pred, labels = _validated_labels(y_true, y_pred, labels)
    matrix = _confusion_matrix_validated(y_true, y_pred, labels)
    total = int(matrix.sum())
    per_class = []
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        recall = _ratio(true_positive, true_positive + false_negative)
        per_class.append(ClassMetrics(
            label=label,
            support=true_positive + false_negative,
            predicted=true_positive + false_positive,
            true_positive=true_positive,
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=_ratio(true_positive, true_positive + false_positive),
            recall=recall,
            f1=_ratio(2 * true_positive,
                      2 * true_positive + false_positive + false_negative),
            tpr=recall,
            fpr=_ratio(false_positive, false_positive + true_negative),
        ))
    per_class = tuple(per_class)
    matrix.setflags(write=False)
    return ClassificationReport(
        labels=labels,
        confusion_matrix=matrix,
        accuracy=float(np.trace(matrix) / total),
        per_class=per_class,
        macro_precision=_macro(item.precision for item in per_class),
        macro_recall=_macro(item.recall for item in per_class),
        macro_f1=_macro(item.f1 for item in per_class),
        macro_tpr=_macro(item.tpr for item in per_class),
        macro_fpr=_macro(item.fpr for item in per_class),
    )

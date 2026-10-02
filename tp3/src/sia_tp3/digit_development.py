"""Construcción de development de dígitos a partir de fuentes solapadas."""

from dataclasses import dataclass

import numpy as np

from .data import _load_digit_file


@dataclass(frozen=True)
class UniqueDigitDevelopment:
    """Unión por imagen, con trazabilidad hacia las filas de ambos CSV."""

    X: np.ndarray
    y: np.ndarray
    primary_rows: np.ndarray
    additional_rows: np.ndarray
    extra_copies: int


def load_unique_digit_development(primary_path, additional_path):
    """Unir dos fuentes eliminando imágenes idénticas.

    Las filas se numeran como en el CSV, comenzando en 2 por el encabezado. Si
    una imagen aparece en ambos archivos se conserva una sola representación y
    se registran las dos filas de origen. Una imagen con etiquetas distintas es
    un conflicto y detiene la construcción.
    """
    primary_X, primary_y = _load_digit_file(primary_path)
    additional_X, additional_y = _load_digit_file(additional_path)
    if primary_X.dtype != additional_X.dtype:
        raise ValueError("las fuentes de dígitos deben usar el mismo dtype")

    images = []
    labels = []
    primary_rows = []
    additional_rows = []
    key_to_index = {}
    extra_copies = 0

    def add_source(source_X, source_y, *, primary):
        nonlocal extra_copies
        contiguous = np.ascontiguousarray(source_X)
        for offset, (image, label) in enumerate(zip(contiguous, source_y)):
            key = image.tobytes()
            csv_row = offset + 2
            if key not in key_to_index:
                key_to_index[key] = len(images)
                images.append(image.copy())
                labels.append(int(label))
                primary_rows.append(csv_row if primary else -1)
                additional_rows.append(-1 if primary else csv_row)
                continue

            index = key_to_index[key]
            if labels[index] != int(label):
                raise ValueError(
                    "una imagen idéntica tiene etiquetas contradictorias: "
                    f"{labels[index]} y {int(label)}")
            extra_copies += 1
            rows = primary_rows if primary else additional_rows
            if rows[index] != -1:
                raise ValueError("una fuente contiene imágenes internas duplicadas")
            rows[index] = csv_row

    add_source(primary_X, primary_y, primary=True)
    add_source(additional_X, additional_y, primary=False)
    return UniqueDigitDevelopment(
        X=np.stack(images),
        y=np.asarray(labels, dtype=np.int64),
        primary_rows=np.asarray(primary_rows, dtype=np.int64),
        additional_rows=np.asarray(additional_rows, dtype=np.int64),
        extra_copies=extra_copies,
    )

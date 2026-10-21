"""Carga y representación bipolar de los códigos QR del ejercicio de Hopfield."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.image as mpimg
import numpy as np


QR_FILENAMES: Tuple[str, ...] = (
    "beverages_menu_qr.png",
    "dessert_menu_qr.png",
    "food_menu_qr.png",
    "specials_menu_qr.png",
    "wine_menu_qr.png",
)
QR_SHAPE = (37, 37)


@dataclass(frozen=True)
class QRPatternSet:
    """Los QR conservando sus formas matricial y vectorial."""

    names: Tuple[str, ...]
    images: np.ndarray
    vectors: np.ndarray


def _binary_grayscale(image: np.ndarray, path: Path) -> np.ndarray:
    """Validar una imagen binaria y llevar sus valores al intervalo 0--1."""

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"{path.name} debe ser una imagen en escala de grises")
    if values.shape != QR_SHAPE:
        raise ValueError(
            f"{path.name} debe medir {QR_SHAPE[0]}x{QR_SHAPE[1]}, "
            f"pero mide {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{path.name} debe contener valores numéricos")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contiene valores no finitos")

    unique = np.unique(values)
    if np.isin(unique, (0, 1)).all():
        return values.astype(np.uint8)
    if np.isin(unique, (0, 255)).all():
        return (values // 255).astype(np.uint8)
    raise ValueError(
        f"{path.name} debe ser binaria: sólo se permiten blanco y negro"
    )


def _load_bipolar_image(path: Path) -> np.ndarray:
    """Cargar un QR asignando +1 al negro y -1 al blanco."""

    grayscale = _binary_grayscale(mpimg.imread(path), path)
    return np.where(grayscale == 0, 1, -1).astype(np.int8)


def load_qr_patterns(directory: Path | str) -> QRPatternSet:
    """Cargar los cinco QR del enunciado en un orden fijo y reproducible."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"no existe el directorio de QR: {root}")

    paths = tuple(root / filename for filename in QR_FILENAMES)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"faltan archivos QR: {missing}")

    images = np.asarray([_load_bipolar_image(path) for path in paths])
    names = tuple(path.stem.removesuffix("_menu_qr") for path in paths)
    vectors = images.reshape(len(images), -1)
    return QRPatternSet(names=names, images=images, vectors=vectors)

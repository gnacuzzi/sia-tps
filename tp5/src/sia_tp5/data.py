"""Leer los sprites binarios de ``font.h`` sin copiar sus datos a mano."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np


FONT_HEIGHT = 7
FONT_WIDTH = 5
FONT_SIZE = FONT_HEIGHT * FONT_WIDTH
EXPECTED_CODEPOINTS = tuple(range(0x60, 0x80))

_SPRITE_LINE = re.compile(
    r"\{\s*((?:0x[0-9a-fA-F]+\s*,?\s*){7})\}\s*,?\s*"
    r"//\s*0x([0-9a-fA-F]+),"
)
_HEX_VALUE = re.compile(r"0x[0-9a-fA-F]+")


@dataclass(frozen=True)
class FontDataset:
    """Conservar juntas las imagenes, los vectores y sus identificadores."""

    images: np.ndarray
    patterns: np.ndarray
    codepoints: Tuple[int, ...]
    labels: Tuple[str, ...]


def row_to_bits(value: int, *, width: int = FONT_WIDTH) -> np.ndarray:
    """Convertir un entero en bits, del bit mas significativo al menos."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("value debe ser un entero")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width debe ser un entero positivo")
    if not 0 <= value < 2 ** width:
        raise ValueError(f"value debe poder representarse con {width} bits")
    return np.array([(value >> shift) & 1 for shift in range(width - 1, -1, -1)],
                    dtype=np.float64)


def _default_font_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "font.h"


def _label(codepoint: int) -> str:
    return "DEL" if codepoint == 0x7F else chr(codepoint)


def load_font(path: Optional[Path] = None) -> FontDataset:
    """Leer los 32 sprites 7x5 y devolver tambien su version aplanada."""
    source = _default_font_path() if path is None else Path(path)
    text = source.read_text(encoding="utf-8")

    codepoints = []
    images = []
    for match in _SPRITE_LINE.finditer(text):
        values_text, codepoint_text = match.groups()
        row_values = [int(value, 16) for value in _HEX_VALUE.findall(values_text)]
        if len(row_values) != FONT_HEIGHT:
            raise ValueError("cada sprite debe contener exactamente siete filas")
        images.append(np.stack([row_to_bits(value) for value in row_values]))
        codepoints.append(int(codepoint_text, 16))

    if tuple(codepoints) != EXPECTED_CODEPOINTS:
        raise ValueError(
            "font.h debe contener, en orden, los 32 caracteres entre 0x60 y 0x7f"
        )

    image_matrix = np.stack(images)
    patterns = image_matrix.reshape(len(image_matrix), FONT_SIZE).copy()
    return FontDataset(
        images=image_matrix,
        patterns=patterns,
        codepoints=tuple(codepoints),
        labels=tuple(_label(codepoint) for codepoint in codepoints),
    )

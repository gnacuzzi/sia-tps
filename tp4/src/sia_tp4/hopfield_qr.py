"""Carga y representación bipolar de los códigos QR del ejercicio de Hopfield."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pca_analysis import _load_pyplot, _save_figure


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

    grayscale = _binary_grayscale(_load_pyplot().imread(path), path)
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


def gram_matrices(patterns: QRPatternSet) -> Tuple[np.ndarray, np.ndarray]:
    """Calcular las matrices de Gram cruda y normalizada por píxel."""

    vectors = patterns.vectors.astype(np.int64)
    gram = vectors @ vectors.T
    normalized = gram / vectors.shape[1]
    return gram, normalized


def pairwise_similarity_rows(patterns: QRPatternSet) -> List[dict]:
    """Describir cada par a partir de su producto interno bipolar."""

    gram, normalized = gram_matrices(patterns)
    pixel_count = patterns.vectors.shape[1]
    rows = []
    for first in range(len(patterns.names)):
        for second in range(first + 1, len(patterns.names)):
            inner_product = int(gram[first, second])
            different_pixels = (pixel_count - inner_product) // 2
            rows.append(
                {
                    "first": patterns.names[first],
                    "second": patterns.names[second],
                    "inner_product": inner_product,
                    "absolute_inner_product": abs(inner_product),
                    "normalized_inner_product": float(normalized[first, second]),
                    "different_pixels": different_pixels,
                    "different_pixel_fraction": different_pixels / pixel_count,
                }
            )
    return rows


def pixel_balance_rows(patterns: QRPatternSet) -> List[dict]:
    """Contar módulos negros y blancos de cada patrón."""

    rows = []
    pixel_count = patterns.vectors.shape[1]
    for name, vector in zip(patterns.names, patterns.vectors):
        black = int(np.count_nonzero(vector == 1))
        rows.append(
            {
                "name": name,
                "black_pixels": black,
                "white_pixels": pixel_count - black,
                "black_fraction": black / pixel_count,
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_orthogonality_tables(output: Path, patterns: QRPatternSet) -> None:
    """Guardar Gram, similitudes por par y balances de píxeles."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    gram, normalized = gram_matrices(patterns)

    matrix_rows = []
    normalized_rows = []
    for name, raw_row, normalized_row in zip(patterns.names, gram, normalized):
        matrix_rows.append(
            {"name": name, **dict(zip(patterns.names, map(int, raw_row)))}
        )
        normalized_rows.append(
            {
                "name": name,
                **dict(zip(patterns.names, map(float, normalized_row))),
            }
        )
    fields = ["name", *patterns.names]
    _write_csv(generated / "matriz_gram.csv", fields, matrix_rows)
    _write_csv(
        generated / "matriz_gram_normalizada.csv", fields, normalized_rows
    )
    pair_rows = pairwise_similarity_rows(patterns)
    _write_csv(generated / "similitud_pares.csv", list(pair_rows[0]), pair_rows)
    balance_rows = pixel_balance_rows(patterns)
    _write_csv(
        generated / "balance_pixeles.csv", list(balance_rows[0]), balance_rows
    )


def write_orthogonality_plot(output: Path, patterns: QRPatternSet) -> None:
    """Graficar la matriz normalizada con sus valores explícitos."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _, normalized = gram_matrices(patterns)
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0)
    indices = np.arange(len(patterns.names))
    axis.set_xticks(indices, patterns.names, rotation=35, ha="right")
    axis.set_yticks(indices, patterns.names)
    axis.set_title("Productos internos normalizados entre QR")
    for row in indices:
        for column in indices:
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Producto interno / 1369")
    figure.tight_layout()
    _save_figure(figure, generated / "matriz_gram_normalizada.svg")
    plt.close(figure)


def write_orthogonality_report(output: Path, patterns: QRPatternSet) -> None:
    """Documentar los resultados descriptivos del análisis."""

    pairs = pairwise_similarity_rows(patterns)
    balance = pixel_balance_rows(patterns)
    similarities = np.asarray(
        [row["normalized_inner_product"] for row in pairs]
    )
    closest = max(pairs, key=lambda row: row["absolute_inner_product"])
    minimum = min(row["absolute_inner_product"] for row in pairs)
    least_similar = [
        row for row in pairs if row["absolute_inner_product"] == minimum
    ]
    least_similar_names = ", ".join(
        f"**{row['first']}–{row['second']}**" for row in least_similar
    )
    report = f"""# Análisis inicial de los QR para Hopfield

## Representación

Los cinco códigos QR se cargan como matrices binarias de 37×37 y se convierten
a representación bipolar: negro `+1` y blanco `-1`. Cada patrón se aplana en
un vector de 1369 componentes. La carga es determinista y no usa semillas ni
hiperparámetros.

## Matriz de Gram

Se calcula `G = XXᵀ` usando enteros de 64 bits y se normaliza dividiendo por
1369. La diagonal normalizada vale 1 porque cada patrón se compara consigo
mismo. Para patrones bipolares, un valor fuera de la diagonal cercano a cero
indicaría mayor cercanía a la ortogonalidad.

- producto interno normalizado medio fuera de la diagonal: **{similarities.mean():.3f}**;
- mínimo absoluto: **{least_similar[0]['normalized_inner_product']:.3f}** en
  {least_similar_names};
- máximo absoluto: **{closest['normalized_inner_product']:.3f}** entre
  **{closest['first']}** y **{closest['second']}**.

![Matriz de Gram normalizada](generated/matriz_gram_normalizada.svg)

Todos los productos internos son positivos y altos. Por lo tanto, estos QR no
son cercanos a ser ortogonales: comparten muchos más píxeles con el mismo signo
que con signo opuesto. Esto anticipa interferencia al almacenar los cinco,
pero todavía no demuestra qué ocurrirá durante la recuperación; eso debe
comprobarse entrenando la red en la etapa siguiente.

## Balance de píxeles

{chr(10).join(f"- **{row['name']}**: {row['black_pixels']} negros, {row['white_pixels']} blancos ({row['black_fraction'] * 100:.1f} % negros)." for row in balance)}

El balance es cercano a 50 % en los cinco casos. Por eso, los productos
internos altos no se explican simplemente por una mayoría global de blanco o
negro, sino por posiciones coincidentes entre los QR, incluidas sus estructuras
compartidas. No se modifica ni se centra la entrada: se conserva la
codificación bipolar pedida para Hopfield.

## Archivos generados

- `generated/matriz_gram.csv`: productos internos crudos;
- `generated/matriz_gram_normalizada.csv`: productos internos divididos por 1369;
- `generated/similitud_pares.csv`: los diez pares y sus píxeles diferentes;
- `generated/balance_pixeles.csv`: cantidad de módulos negros y blancos;
- `generated/matriz_gram_normalizada.svg`: visualización única del Gram.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/hopfield"),
        help="Directorio que contiene los cinco QR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/hopfield-qr"),
        help="Directorio del informe reproducible.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    patterns = load_qr_patterns(args.data)
    write_orthogonality_tables(args.output, patterns)
    write_orthogonality_plot(args.output, patterns)
    write_orthogonality_report(args.output, patterns)
    similarities = [
        row["normalized_inner_product"]
        for row in pairwise_similarity_rows(patterns)
    ]
    print(
        f"Informe generado en {args.output}; "
        f"similitud media={np.mean(similarities):.3f}; "
        f"máxima={np.max(np.abs(similarities)):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

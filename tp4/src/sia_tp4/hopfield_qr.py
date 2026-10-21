"""Carga y representación bipolar de los códigos QR del ejercicio de Hopfield."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .hopfield import HopfieldModel, RecallResult, fit_hopfield
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


def _recall_outcome(
    model: HopfieldModel,
    target_index: int,
    result: RecallResult,
) -> Tuple[str, Optional[int]]:
    """Clasificar el destino de una consulta respecto de los patrones guardados."""

    if result.status == "cycle":
        return "cycle", None
    if result.status == "max_iterations":
        return "max_iterations", None
    matched_index = model.matching_pattern_index(result.final_state)
    if matched_index == target_index:
        return "correct", matched_index
    if matched_index is not None:
        return "other_stored", matched_index
    return "spurious_fixed", None


def run_zero_noise_recall(
    patterns: QRPatternSet,
    max_iterations: int = 100,
) -> Tuple[List[dict], List[dict]]:
    """Consultar cada QR original en una red que almacena los cinco juntos."""

    model = fit_hopfield(patterns.vectors)
    rows = []
    cases = []
    for target_index, (name, target) in enumerate(
        zip(patterns.names, patterns.vectors)
    ):
        result = model.recall(target, max_iterations=max_iterations)
        outcome, matched_index = _recall_outcome(
            model, target_index, result
        )
        final_error = int(np.count_nonzero(result.final_state != target))
        remains_stable = bool(
            result.status == "fixed_point"
            and len(result.states) >= 2
            and np.array_equal(result.states[0], result.states[1])
        )
        row = {
            "target": name,
            "remains_stable": remains_stable,
            "outcome": outcome,
            "status": result.status,
            "matched_pattern": (
                patterns.names[matched_index]
                if matched_index is not None
                else ""
            ),
            "iterations": result.iterations,
            "cycle_length": result.cycle_length or "",
            "final_error_pixels": final_error,
            "final_error_fraction": final_error / len(target),
            "initial_energy": float(result.energies[0]),
            "final_energy": float(result.energies[-1]),
            "energy_change": float(
                result.energies[-1] - result.energies[0]
            ),
        }
        rows.append(row)
        cases.append(
            {
                "target_index": target_index,
                "target": target,
                "row": row,
                "result": result,
            }
        )
    return rows, cases


def energy_trajectory_rows(cases: Sequence[dict]) -> List[dict]:
    """Convertir las trayectorias de energía en una tabla larga."""

    rows = []
    for case in cases:
        target = case["target"]
        name = case["row"]["target"]
        result = case["result"]
        for iteration, (state, energy) in enumerate(
            zip(result.states, result.energies)
        ):
            rows.append(
                {
                    "target": name,
                    "iteration": iteration,
                    "energy": float(energy),
                    "error_pixels": int(np.count_nonzero(state != target)),
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


def write_zero_noise_tables(
    output: Path,
    rows: Sequence[dict],
    cases: Sequence[dict],
) -> None:
    """Guardar el resumen y la energía completa de las consultas sin ruido."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "recuperacion_sin_ruido.csv", list(rows[0]), rows)
    energy_rows = energy_trajectory_rows(cases)
    _write_csv(
        generated / "energia_sin_ruido.csv",
        list(energy_rows[0]),
        energy_rows,
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


def write_zero_noise_plots(output: Path, cases: Sequence[dict]) -> None:
    """Comparar objetivos, consultas y destinos, y mostrar sus energías."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()

    figure, axes = plt.subplots(
        len(cases), 4, figsize=(9.4, 2.15 * len(cases)), squeeze=False
    )
    titles = ("Objetivo", "Consulta S(0)", "Estado final", "Diferencias")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row_index, case in enumerate(cases):
        target = case["target"].reshape(QR_SHAPE)
        initial = case["result"].states[0].reshape(QR_SHAPE)
        final = case["result"].final_state.reshape(QR_SHAPE)
        difference = target != final
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].imshow(target, cmap="binary", vmin=-1, vmax=1)
        axes[row_index, 1].imshow(initial, cmap="binary", vmin=-1, vmax=1)
        axes[row_index, 2].imshow(final, cmap="binary", vmin=-1, vmax=1)
        axes[row_index, 3].imshow(
            difference, cmap="Reds", vmin=0, vmax=1
        )
        row = case["row"]
        axes[row_index, 0].set_ylabel(
            f"{row['target']}\n{row['outcome']} · "
            f"{row['final_error_pixels']} errores",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            fontsize=8,
        )
    figure.suptitle("Recuperación de los cinco QR sin ruido", y=1.0)
    figure.tight_layout()
    _save_figure(figure, generated / "recuperacion_sin_ruido.svg")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), squeeze=False)
    for axis, case in zip(axes.reshape(-1), cases):
        energies = case["result"].energies
        iterations = np.arange(len(energies))
        axis.plot(iterations, energies, marker="o", color="#3b82f6")
        axis.set_title(case["row"]["target"])
        axis.set_xlabel("Iteración")
        axis.set_ylabel("Energía")
        axis.set_xticks(iterations)
        axis.margins(x=0.12, y=0.15)
        axis.grid(alpha=0.25)
    for axis in axes.reshape(-1)[len(cases) :]:
        axis.axis("off")
    figure.suptitle("Energía durante la recuperación sin ruido")
    figure.tight_layout()
    _save_figure(figure, generated / "energia_sin_ruido.svg")
    plt.close(figure)


def write_report(
    output: Path,
    patterns: QRPatternSet,
    zero_noise_rows: Sequence[dict],
    zero_noise_cases: Sequence[dict],
    max_iterations: int,
) -> None:
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
    unique_final_states = {
        case["result"].final_state.tobytes() for case in zero_noise_cases
    }
    correct_count = sum(
        row["outcome"] == "correct" for row in zero_noise_rows
    )
    stable_count = sum(row["remains_stable"] for row in zero_noise_rows)
    transitions = ", ".join(
        f"**{row['target']}→{row['matched_pattern'] or 'espurio'}**"
        for row in zero_noise_rows
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
que con signo opuesto. Esta observación anticipa interferencia al almacenar los
cinco, pero por sí sola no demuestra el comportamiento dinámico; la prueba de
recuperación que aparece debajo comprueba esa hipótesis.

## Balance de píxeles

{chr(10).join(f"- **{row['name']}**: {row['black_pixels']} negros, {row['white_pixels']} blancos ({row['black_fraction'] * 100:.1f} % negros)." for row in balance)}

El balance es cercano a 50 % en los cinco casos. Por eso, los productos
internos altos no se explican simplemente por una mayoría global de blanco o
negro, sino por posiciones coincidentes entre los QR, incluidas sus estructuras
compartidas. No se modifica ni se centra la entrada: se conserva la
codificación bipolar pedida para Hopfield.

## Recuperación sin ruido

Se entrenó una única red mediante la regla de Hebb almacenando simultáneamente
los cinco vectores. Luego cada QR original se usó como consulta, sin modificar
ningún píxel. El límite de {max_iterations} iteraciones funciona sólo como
resguardo y no fue alcanzado. Esta prueba tampoco usa semillas.

- patrones que permanecieron estables: **{stable_count}/5**;
- recuperaciones correctas: **{correct_count}/5**;
- estados finales distintos encontrados: **{len(unique_final_states)}**;
- destinos: {transitions}.

La primera actualización lleva cada consulta al estado espurio y la segunda
confirma que ese estado no cambia; por eso todas las filas registran dos
iteraciones.

{chr(10).join(f"- **{row['target']}**: `{row['outcome']}`, {row['iterations']} iteraciones, {row['final_error_pixels']} píxeles de error, energía {row['initial_energy']:.1f} → {row['final_energy']:.1f}." for row in zero_noise_rows)}

En los cinco recorridos la energía disminuye durante la primera actualización
y permanece constante al confirmar el punto fijo. Esta observación describe
estas trayectorias concretas; no se usa para cambiar la regla de actualización.

![Recuperación sin ruido](generated/recuperacion_sin_ruido.svg)

![Energía sin ruido](generated/energia_sin_ruido.svg)

El chequeo reproducible confirma que ninguno de los QR originales es un punto
fijo cuando se almacenan los cinco juntos. Todos convergen al mismo estado
fijo espurio. Por lo tanto, un barrido de ruido sobre esta red no puede
interpretarse aisladamente como robustez frente a perturbaciones: ya existe
interferencia con 0 % de ruido. El paso siguiente debe separar este efecto
estudiando subconjuntos de uno a cinco patrones.

## Archivos generados

- `generated/matriz_gram.csv`: productos internos crudos;
- `generated/matriz_gram_normalizada.csv`: productos internos divididos por 1369;
- `generated/similitud_pares.csv`: los diez pares y sus píxeles diferentes;
- `generated/balance_pixeles.csv`: cantidad de módulos negros y blancos;
- `generated/recuperacion_sin_ruido.csv`: resultado de cada consulta original;
- `generated/energia_sin_ruido.csv`: energía y error en cada iteración;
- `generated/matriz_gram_normalizada.svg`: visualización del Gram;
- `generated/recuperacion_sin_ruido.svg`: objetivos, consultas y estados finales;
- `generated/energia_sin_ruido.svg`: trayectorias de energía.
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
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Límite de seguridad para cada recuperación (por defecto, 100).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    patterns = load_qr_patterns(args.data)
    zero_noise_rows, zero_noise_cases = run_zero_noise_recall(
        patterns, max_iterations=args.max_iterations
    )
    write_orthogonality_tables(args.output, patterns)
    write_zero_noise_tables(args.output, zero_noise_rows, zero_noise_cases)
    write_orthogonality_plot(args.output, patterns)
    write_zero_noise_plots(args.output, zero_noise_cases)
    write_report(
        args.output,
        patterns,
        zero_noise_rows,
        zero_noise_cases,
        args.max_iterations,
    )
    similarities = [
        row["normalized_inner_product"]
        for row in pairwise_similarity_rows(patterns)
    ]
    print(
        f"Informe generado en {args.output}; "
        f"similitud media={np.mean(similarities):.3f}; "
        f"máxima={np.max(np.abs(similarities)):.3f}; "
        f"recuperaciones sin ruido="
        f"{sum(row['outcome'] == 'correct' for row in zero_noise_rows)}/5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

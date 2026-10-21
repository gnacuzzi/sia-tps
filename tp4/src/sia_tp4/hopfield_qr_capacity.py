"""Barrido exhaustivo de capacidad de Hopfield con los cinco QR."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .hopfield import fit_hopfield
from .hopfield_qr import QRPatternSet, load_qr_patterns
from .pca_analysis import _load_pyplot, _save_figure


@dataclass(frozen=True)
class QRCapacityDesign:
    """Único resguardo numérico del barrido determinista sin ruido."""

    max_iterations: int


def load_capacity_design(path: Path | str) -> QRCapacityDesign:
    """Leer una configuración mínima y rechazar campos no utilizados."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {"max_iterations"}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"campos faltantes: {missing}; desconocidos: {unknown}")
    maximum = raw["max_iterations"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_iterations debe ser un entero positivo")
    return QRCapacityDesign(max_iterations=maximum)


def qr_subsets(pattern_count: int) -> List[Tuple[int, ...]]:
    """Enumerar todos los subconjuntos no vacíos en un orden fijo."""

    if pattern_count < 1:
        raise ValueError("pattern_count debe ser positivo")
    return [
        subset
        for size in range(1, pattern_count + 1)
        for subset in combinations(range(pattern_count), size)
    ]


def _gram_statistics(vectors: np.ndarray) -> dict:
    """Resumir el valor absoluto de los cruces de un subconjunto."""

    size, pixel_count = vectors.shape
    if size == 1:
        return {
            "pair_count": 0,
            "mean_absolute_inner_product": "",
            "max_absolute_inner_product": "",
            "mean_absolute_normalized_inner_product": "",
            "max_absolute_normalized_inner_product": "",
        }
    integer_vectors = vectors.astype(np.int64)
    gram = integer_vectors @ integer_vectors.T
    values = np.abs(gram[np.triu_indices(size, k=1)])
    return {
        "pair_count": len(values),
        "mean_absolute_inner_product": float(np.mean(values)),
        "max_absolute_inner_product": int(np.max(values)),
        "mean_absolute_normalized_inner_product": float(
            np.mean(values) / pixel_count
        ),
        "max_absolute_normalized_inner_product": float(
            np.max(values) / pixel_count
        ),
    }


def run_capacity_sweep(
    patterns: QRPatternSet,
    max_iterations: int = 100,
) -> Tuple[List[dict], List[dict]]:
    """Consultar los originales en cada subconjunto no vacío de los QR."""

    query_rows: List[dict] = []
    subset_rows: List[dict] = []
    for subset_number, indices in enumerate(
        qr_subsets(len(patterns.names)), start=1
    ):
        subset_id = f"S{subset_number:02d}"
        names = tuple(patterns.names[index] for index in indices)
        stored = patterns.vectors[list(indices)]
        model = fit_hopfield(stored)
        subset_queries = []
        for local_index, global_index in enumerate(indices):
            target = patterns.vectors[global_index]
            result = model.recall(target, max_iterations=max_iterations)
            matched_index = model.matching_pattern_index(result.final_state)
            remains_stable = bool(
                result.status == "fixed_point"
                and len(result.states) >= 2
                and np.array_equal(result.states[0], result.states[1])
            )
            if result.status == "cycle":
                outcome = "cycle"
            elif result.status == "max_iterations":
                outcome = "max_iterations"
            elif matched_index == local_index:
                outcome = "correct"
            elif matched_index is not None:
                outcome = "other_stored"
            else:
                outcome = "spurious_fixed"
            final_error = int(np.count_nonzero(result.final_state != target))
            row = {
                "subset_id": subset_id,
                "subset_size": len(indices),
                "stored_patterns": "+".join(names),
                "target": patterns.names[global_index],
                "remains_stable": remains_stable,
                "outcome": outcome,
                "status": result.status,
                "matched_pattern": (
                    names[matched_index] if matched_index is not None else ""
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
                "energy_non_increasing": bool(
                    np.all(np.diff(result.energies) <= 1e-9)
                ),
            }
            query_rows.append(row)
            subset_queries.append(row)

        stable_count = sum(row["remains_stable"] for row in subset_queries)
        correct_count = sum(
            row["outcome"] == "correct" for row in subset_queries
        )
        subset_rows.append(
            {
                "subset_id": subset_id,
                "subset_size": len(indices),
                "stored_patterns": "+".join(names),
                **_gram_statistics(stored),
                "stable_count": stable_count,
                "stable_rate": stable_count / len(indices),
                "fully_stable": stable_count == len(indices),
                "correct_count": correct_count,
                "correct_rate": correct_count / len(indices),
                "other_stored_count": sum(
                    row["outcome"] == "other_stored"
                    for row in subset_queries
                ),
                "spurious_fixed_count": sum(
                    row["outcome"] == "spurious_fixed"
                    for row in subset_queries
                ),
                "cycle_count": sum(
                    row["outcome"] == "cycle" for row in subset_queries
                ),
                "max_iterations_count": sum(
                    row["outcome"] == "max_iterations"
                    for row in subset_queries
                ),
                "mean_final_error_pixels": float(
                    np.mean(
                        [row["final_error_pixels"] for row in subset_queries]
                    )
                ),
                "max_final_error_pixels": max(
                    row["final_error_pixels"] for row in subset_queries
                ),
                "mean_iterations": float(
                    np.mean([row["iterations"] for row in subset_queries])
                ),
                "all_energy_non_increasing": all(
                    row["energy_non_increasing"] for row in subset_queries
                ),
            }
        )
    return query_rows, subset_rows


def summarize_capacity_by_size(subset_rows: Sequence[dict]) -> List[dict]:
    """Agregar el barrido por cantidad almacenada, sin introducir semillas."""

    rows = []
    for size in sorted({row["subset_size"] for row in subset_rows}):
        selected = [row for row in subset_rows if row["subset_size"] == size]
        subset_count = len(selected)
        query_count = sum(row["subset_size"] for row in selected)
        stable_queries = sum(row["stable_count"] for row in selected)
        correct_queries = sum(row["correct_count"] for row in selected)
        fully_stable = sum(row["fully_stable"] for row in selected)
        pairwise = [row for row in selected if row["pair_count"] > 0]
        rows.append(
            {
                "subset_size": size,
                "subset_count": subset_count,
                "fully_stable_subset_count": fully_stable,
                "fully_stable_subset_rate": fully_stable / subset_count,
                "query_count": query_count,
                "stable_query_count": stable_queries,
                "stable_query_rate": stable_queries / query_count,
                "correct_query_count": correct_queries,
                "correct_query_rate": correct_queries / query_count,
                "min_mean_absolute_normalized_inner_product": (
                    min(
                        row["mean_absolute_normalized_inner_product"]
                        for row in pairwise
                    )
                    if pairwise
                    else ""
                ),
                "max_mean_absolute_normalized_inner_product": (
                    max(
                        row["mean_absolute_normalized_inner_product"]
                        for row in pairwise
                    )
                    if pairwise
                    else ""
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_capacity_tables(
    output: Path,
    query_rows: Sequence[dict],
    subset_rows: Sequence[dict],
    size_rows: Sequence[dict],
) -> None:
    """Persistir resultados completos y los dos niveles de agregación."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "consultas_por_subconjunto.csv", query_rows)
    _write_csv(generated / "resumen_por_subconjunto.csv", subset_rows)
    _write_csv(generated / "resumen_por_tamano.csv", size_rows)


def write_capacity_plots(
    output: Path,
    subset_rows: Sequence[dict],
    size_rows: Sequence[dict],
) -> None:
    """Visualizar capacidad por tamaño, por subconjunto y por interferencia."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()

    sizes = np.asarray([row["subset_size"] for row in size_rows])
    stable_percent = np.asarray(
        [row["fully_stable_subset_rate"] * 100 for row in size_rows]
    )
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    bars = axis.bar(sizes, stable_percent, color="#2563eb", width=0.62)
    for bar, row in zip(bars, size_rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + 3,
            f"{row['fully_stable_subset_count']}/{row['subset_count']}",
            ha="center",
            va="bottom",
        )
    axis.set_xlabel("Cantidad de QR almacenados")
    axis.set_ylabel("Subconjuntos completamente estables (%)")
    axis.set_xticks(sizes)
    axis.set_ylim(0, 112)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, generated / "capacidad_por_tamano.svg")
    plt.close(figure)

    labels = [
        f"{row['subset_size']} · {row['stored_patterns']}"
        for row in subset_rows
    ]
    rates = np.asarray([row["stable_rate"] * 100 for row in subset_rows])
    colors = [
        "#16a34a" if row["fully_stable"] else "#dc2626"
        for row in subset_rows
    ]
    positions = np.arange(len(subset_rows))
    figure, axis = plt.subplots(figsize=(10.5, 10.5))
    axis.barh(positions, rates, color=colors, height=0.72)
    axis.scatter(rates, positions, color=colors, s=24, zorder=3)
    for position, rate in zip(positions, rates):
        axis.text(
            rate + 1.2,
            position,
            f"{rate:.0f} %",
            va="center",
            fontsize=7,
        )
    axis.set_yticks(positions, labels, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlim(-2, 108)
    axis.set_xlabel("Patrones originales que permanecen estables (%)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, generated / "estabilidad_por_subconjunto.svg")
    plt.close(figure)

    comparable = [row for row in subset_rows if row["pair_count"] > 0]
    color_by_size = {
        2: "#16a34a",
        3: "#f59e0b",
        4: "#dc2626",
        5: "#7c3aed",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fields = (
        ("mean_absolute_normalized_inner_product", "Cruce absoluto medio"),
        ("max_absolute_normalized_inner_product", "Peor cruce absoluto"),
    )
    for axis, (field, label) in zip(axes, fields):
        for size in range(2, 6):
            selected = [
                row for row in comparable if row["subset_size"] == size
            ]
            axis.scatter(
                [row[field] for row in selected],
                [row["stable_rate"] * 100 for row in selected],
                s=55,
                alpha=0.8,
                label=f"{size} QR",
                color=color_by_size[size],
                edgecolor="white",
                linewidth=0.5,
            )
        axis.set_xlabel(label)
        axis.set_ylim(-5, 105)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Patrones originales estables (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    _save_figure(figure, generated / "estabilidad_vs_interferencia.svg")
    plt.close(figure)


def write_capacity_report(
    output: Path,
    patterns: QRPatternSet,
    design: QRCapacityDesign,
    query_rows: Sequence[dict],
    subset_rows: Sequence[dict],
    size_rows: Sequence[dict],
) -> None:
    """Redactar diseño, resultados y límites del barrido exhaustivo."""

    size_lines = [
        "| QR almacenados | Subconjuntos estables | Consultas estables |",
        "|---:|---:|---:|",
    ]
    for row in size_rows:
        size_lines.append(
            f"| {row['subset_size']} | "
            f"{row['fully_stable_subset_count']}/{row['subset_count']} "
            f"({row['fully_stable_subset_rate'] * 100:.0f} %) | "
            f"{row['stable_query_count']}/{row['query_count']} "
            f"({row['stable_query_rate'] * 100:.0f} %) |"
        )
    valid = [row for row in subset_rows if row["fully_stable"]]
    largest_any = max(row["subset_size"] for row in valid)
    largest_all = max(
        row["subset_size"]
        for row in size_rows
        if row["fully_stable_subset_count"] == row["subset_count"]
    )
    pair_rows = [row for row in subset_rows if row["subset_size"] == 2]
    most_correlated_pair = max(
        pair_rows,
        key=lambda row: row["max_absolute_normalized_inner_product"],
    )
    energy_violations = sum(
        not row["energy_non_increasing"] for row in query_rows
    )
    report = f"""# Capacidad de Hopfield con subconjuntos de QR

## Diseño experimental

Se enumeran exhaustivamente los {len(subset_rows)} subconjuntos no vacíos de
los cinco QR: cinco individuales, diez pares, diez ternas, cinco conjuntos de
cuatro y el conjunto completo. Para cada subconjunto se construye una nueva red
con la misma regla de Hebb y la misma actualización sincrónica usadas en el
experimento principal. Luego se consulta cada patrón almacenado sin modificarlo.

El experimento es determinista: no agrega ruido, no inicializa pesos al azar y
no necesita semillas. El límite de {design.max_iterations} iteraciones es sólo
un resguardo de ejecución. En total se evalúan {len(query_rows)} consultas
originales. Un subconjunto se considera completamente estable únicamente si
todos sus patrones son puntos fijos desde la primera actualización.

Para cada combinación también se calculan la media y el máximo de los productos
internos absolutos normalizados entre pares. En los conjuntos de un único QR no
hay productos fuera de la diagonal, por lo que esas celdas quedan vacías.

## Resultado

{chr(10).join(size_lines)}

![Capacidad por tamaño](generated/capacidad_por_tamano.svg)

El corte es exacto para estos cinco datos: todos los patrones individuales y
los diez pares son completamente estables, mientras que ninguna de las diez
ternas ni de las combinaciones mayores conserva un patrón original como punto
fijo. El mayor tamaño para el que funciona **algún** subconjunto es
**{largest_any}** y el mayor tamaño para el que funcionan **todos** es
**{largest_all}**. En este caso ambos criterios coinciden.

Esto permite decir que la capacidad efectiva observada es de dos QR para esta
familia concreta, con esta codificación, regla de aprendizaje y dinámica. No es
una capacidad universal de una red de {patterns.vectors.shape[1]} neuronas.

![Estabilidad por subconjunto](generated/estabilidad_por_subconjunto.svg)

## Relación con la interferencia

El par con mayor producto interno absoluto es
**{most_correlated_pair['stored_patterns']}**, con valor normalizado
{most_correlated_pair['max_absolute_normalized_inner_product']:.3f}, y aun así
sus dos patrones son estables. Al mismo tiempo, todas las ternas fallan, incluso
las que tienen cruces medios menores. En este conjunto, el peor par aislado no
alcanza para explicar el fallo: al agregar un tercer patrón se acumulan términos
cruzados en la matriz de pesos y desaparecen los puntos fijos deseados.

![Estabilidad frente a interferencia](generated/estabilidad_vs_interferencia.svg)

El gráfico permite inspeccionar todos los subconjuntos y no debe interpretarse
como una relación causal estimada con sólo cinco QR. La hipótesis inicial de que
los subconjuntos menos correlacionados serían necesariamente más estables no se
puede sostener dentro de cada tamaño: aquí la separación observada corresponde
por completo a almacenar hasta dos patrones o almacenar tres o más.

## Control de la dinámica

Hubo **{energy_violations}** consultas con algún aumento de energía entre
actualizaciones. Las tablas conservan estado final, cantidad de iteraciones,
error de píxeles y energía para cada consulta, de modo que el criterio de
capacidad no depende solamente del gráfico agregado.

Los {len(valid)} subconjuntos completamente estables —los cinco individuales y
los diez pares— se usan en el
[barrido de robustez con ruido](../hopfield-qr-robustez/README.md). Los
restantes siguen documentados como evidencia de la limitación de capacidad,
pero agregarles ruido no mediría recuperación alrededor de memorias estables.

## Archivos generados

- `generated/consultas_por_subconjunto.csv`: las {len(query_rows)} consultas;
- `generated/resumen_por_subconjunto.csv`: métricas de las {len(subset_rows)} combinaciones;
- `generated/resumen_por_tamano.csv`: agregación determinista por cantidad;
- `generated/capacidad_por_tamano.svg`: proporción de conjuntos estables;
- `generated/estabilidad_por_subconjunto.svg`: resultado de cada combinación;
- `generated/estabilidad_vs_interferencia.svg`: comparación con el Gram.
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
        "--config",
        type=Path,
        default=Path("configs/hopfield/qr_capacity.json"),
        help="Configuración del barrido determinista.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/hopfield-qr-capacidad"),
        help="Directorio del informe reproducible.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    design = load_capacity_design(args.config)
    patterns = load_qr_patterns(args.data)
    query_rows, subset_rows = run_capacity_sweep(
        patterns, max_iterations=design.max_iterations
    )
    size_rows = summarize_capacity_by_size(subset_rows)
    write_capacity_tables(args.output, query_rows, subset_rows, size_rows)
    write_capacity_plots(args.output, subset_rows, size_rows)
    write_capacity_report(
        args.output,
        patterns,
        design,
        query_rows,
        subset_rows,
        size_rows,
    )
    valid_count = sum(row["fully_stable"] for row in subset_rows)
    print(
        f"Informe generado en {args.output}; "
        f"subconjuntos={len(subset_rows)}; consultas={len(query_rows)}; "
        f"completamente estables={valid_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

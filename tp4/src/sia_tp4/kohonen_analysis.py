"""Ejecutar y documentar configuraciones de Kohonen sobre los creadores."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen import KohonenConfig, KohonenModel, fit_kohonen
from .pca_analysis import _load_pyplot, _save_figure


_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CONFIG_FIELDS = {
    "grid_size",
    "epochs",
    "learning_rate_start",
    "learning_rate_end",
    "radius_start",
    "radius_end",
    "initialization",
    "shuffle",
    "seed",
}


@dataclass(frozen=True)
class KohonenRun:
    """Una configuración identificable dentro del experimento."""

    name: str
    config: KohonenConfig


@dataclass(frozen=True)
class KohonenRunResult:
    """Modelo y diagnósticos descriptivos de una corrida."""

    run: KohonenRun
    model: KohonenModel
    assignments: np.ndarray
    hit_map: np.ndarray
    u_matrix: np.ndarray
    quantization_error: float

    @property
    def occupied_units(self) -> int:
        return int(np.count_nonzero(self.hit_map))

    @property
    def dead_units(self) -> int:
        return int(self.hit_map.size - self.occupied_units)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/creadores.csv"),
        help="CSV fuente (por defecto, datos/creadores.csv).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kohonen/base.json"),
        help="Configuración de una corrida (por defecto, configs/kohonen/base.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directorio del informe (por defecto, "
            "docs/kohonen-creadores/<nombre-de-corrida>)."
        ),
    )
    return parser.parse_args(argv)


def load_run(path: Path | str) -> KohonenRun:
    """Leer una única configuración y validarla antes de entrenar."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido en {path}: {error.msg}") from error
    if not isinstance(document, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {"name", *_CONFIG_FIELDS}
    unknown = set(document) - expected
    missing = expected - set(document)
    if unknown or missing:
        raise ValueError(
            "campos inválidos en la configuración: "
            f"faltan={sorted(missing)}, sobran={sorted(unknown)}"
        )
    name = document["name"]
    if not isinstance(name, str) or not _RUN_NAME.fullmatch(name):
        raise ValueError("el nombre debe usar minúsculas, números y guiones")
    parameters = {key: document[key] for key in _CONFIG_FIELDS}
    try:
        config = KohonenConfig(**parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(f"configuración inválida en '{name}': {error}") from error
    return KohonenRun(name=name, config=config)


def run_kohonen(
    dataset: CreatorDataset, run: KohonenRun
) -> KohonenRunResult:
    """Entrenar la configuración sobre las filas y variables seleccionadas."""

    model = fit_kohonen(dataset.values, dataset.feature_names, run.config)
    return KohonenRunResult(
        run=run,
        model=model,
        assignments=model.map_samples(dataset.values),
        hit_map=model.hit_map(dataset.values),
        u_matrix=model.u_matrix(),
        quantization_error=model.quantization_error(dataset.values),
    )


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _matrix_rows(matrix: np.ndarray) -> Iterable[dict]:
    columns = [f"column_{column}" for column in range(matrix.shape[1])]
    for row_index, row in enumerate(matrix):
        yield {"row": row_index, **dict(zip(columns, row))}


def write_tables(
    output: Path,
    dataset: CreatorDataset,
    result: KohonenRunResult,
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    _write_csv(
        generated / "resumen.csv",
        [
            "run",
            "grid_size",
            "epochs",
            "learning_rate_start",
            "learning_rate_end",
            "radius_start",
            "radius_end",
            "initialization",
            "shuffle",
            "seed",
            "occupied_units",
            "dead_units",
            "maximum_hits",
            "quantization_error",
            "mean_u_distance",
        ],
        [
            {
                "run": result.run.name,
                **vars(result.run.config),
                "occupied_units": result.occupied_units,
                "dead_units": result.dead_units,
                "maximum_hits": int(result.hit_map.max()),
                "quantization_error": result.quantization_error,
                "mean_u_distance": float(result.u_matrix.mean()),
            }
        ],
    )

    _write_csv(
        generated / "estandarizacion.csv",
        ["variable", "mean", "sample_standard_deviation"],
        (
            {
                "variable": name,
                "mean": mean,
                "sample_standard_deviation": deviation,
            }
            for name, mean, deviation in zip(
                result.model.feature_names,
                result.model.standardizer.means,
                result.model.standardizer.standard_deviations,
            )
        ),
    )

    size = result.run.config.grid_size
    matrix_fields = ["row", *(f"column_{index}" for index in range(size))]
    _write_csv(
        generated / "mapa_hits.csv",
        matrix_fields,
        _matrix_rows(result.hit_map),
    )
    _write_csv(
        generated / "matriz_u.csv",
        matrix_fields,
        _matrix_rows(result.u_matrix),
    )
    _write_csv(
        generated / "asignaciones.csv",
        ["creator_id", "name", "followers", "row", "column"],
        (
            {
                "creator_id": creator_id,
                "name": name,
                "followers": followers,
                "row": int(position[0]),
                "column": int(position[1]),
            }
            for creator_id, name, followers, position in zip(
                dataset.creator_ids,
                dataset.names,
                dataset.followers,
                result.assignments,
            )
        ),
    )
    prototypes = result.model.prototypes(original_scale=True)
    _write_csv(
        generated / "prototipos.csv",
        ["row", "column", *dataset.feature_names],
        (
            {
                "row": row,
                "column": column,
                **dict(zip(dataset.feature_names, prototypes[row, column])),
            }
            for row in range(size)
            for column in range(size)
        ),
    )


def _heatmap(
    plt,
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    value_format: str,
):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="viridis", aspect="equal")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], value_format),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > np.nanmean(matrix) else "black",
                fontsize=8,
            )
    axis.set(
        title=title,
        xlabel="Columna de la grilla",
        ylabel="Fila de la grilla",
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
    )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    return figure


def _creator_grid(plt, dataset: CreatorDataset, result: KohonenRunResult):
    size = result.run.config.grid_size
    cells = [[[] for _ in range(size)] for _ in range(size)]
    for creator_id, (row, column) in zip(
        dataset.creator_ids, result.assignments
    ):
        cells[row][column].append(creator_id)

    figure, axis = plt.subplots(figsize=(max(8, size * 2), max(7, size * 1.7)))
    axis.imshow(result.hit_map, cmap="Blues", aspect="equal")
    for row in range(size):
        for column in range(size):
            label = "\n".join(cells[row][column]) if cells[row][column] else "—"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=max(5, 9 - size),
            )
    axis.set(
        title=f"Creadores asignados — {result.run.name}",
        xlabel="Columna de la grilla",
        ylabel="Fila de la grilla",
        xticks=np.arange(size),
        yticks=np.arange(size),
    )
    return figure


def _component_planes(plt, result: KohonenRunResult):
    feature_names = result.model.feature_names
    column_count = 3
    row_count = int(np.ceil(len(feature_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(column_count * 5, row_count * 4),
        squeeze=False,
    )
    for axis, feature_name in zip(axes.flat, feature_names):
        plane = result.model.component_plane(feature_name, original_scale=True)
        image = axis.imshow(plane, cmap="RdYlBu_r", aspect="equal")
        axis.set_title(feature_name, fontsize=9)
        axis.set_xticks(np.arange(plane.shape[1]))
        axis.set_yticks(np.arange(plane.shape[0]))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(feature_names) :]:
        axis.set_visible(False)
    figure.suptitle(f"Mapas por variable — {result.run.name}")
    figure.tight_layout()
    return figure


def _followers_map(
    dataset: CreatorDataset, result: KohonenRunResult
) -> np.ndarray:
    size = result.run.config.grid_size
    totals = np.zeros((size, size), dtype=np.float64)
    counts = np.zeros((size, size), dtype=np.int64)
    for followers, (row, column) in zip(dataset.followers, result.assignments):
        totals[row, column] += followers
        counts[row, column] += 1
    averages = np.full((size, size), np.nan)
    np.divide(totals, counts, out=averages, where=counts > 0)
    return averages


def _followers_figure(plt, dataset: CreatorDataset, result: KohonenRunResult):
    averages = _followers_map(dataset, result)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        np.ma.masked_invalid(averages), cmap="magma", aspect="equal"
    )
    for row in range(averages.shape[0]):
        for column in range(averages.shape[1]):
            value = averages[row, column]
            axis.text(
                column,
                row,
                "—" if np.isnan(value) else f"{value:,.0f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set(
        title=f"Followers promedio por neurona — {result.run.name}",
        xlabel="Columna de la grilla",
        ylabel="Fila de la grilla",
        xticks=np.arange(averages.shape[1]),
        yticks=np.arange(averages.shape[0]),
    )
    figure.colorbar(image, ax=axis, label="Followers promedio (auxiliar)")
    return figure


def write_plots(
    output: Path,
    dataset: CreatorDataset,
    result: KohonenRunResult,
) -> None:
    plt = _load_pyplot()
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    figures = (
        (
            _heatmap(
                plt,
                result.hit_map,
                f"Mapa de hits — {result.run.name}",
                "Cantidad de creadores",
                "d",
            ),
            "mapa_hits.svg",
        ),
        (
            _heatmap(
                plt,
                result.u_matrix,
                f"Matriz U — {result.run.name}",
                "Distancia euclídea media",
                ".2f",
            ),
            "matriz_u.svg",
        ),
        (_creator_grid(plt, dataset, result), "creadores.svg"),
        (_component_planes(plt, result), "mapas_variables.svg"),
        (_followers_figure(plt, dataset, result), "followers.svg"),
    )
    for figure, filename in figures:
        _save_figure(figure, generated / filename)
        plt.close(figure)


def write_report(output: Path, result: KohonenRunResult) -> None:
    config = result.run.config

    report = f"""# Análisis de la red de Kohonen

Esta corrida usa los mismos creadores y las 15 variables seleccionadas, con
z-score por variable. La BMU se obtiene por distancia euclídea.

## Configuración

| Corrida | Grilla | Épocas | Tasa | Radio | Inicialización | Semilla | Error de cuantización | Ocupadas | Muertas | Máximo de hits |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `{result.run.name}` | {config.grid_size}x{config.grid_size} | {config.epochs} | {config.learning_rate_start:g} → {config.learning_rate_end:g} | {config.radius_start:g} → {config.radius_end:g} | {config.initialization} | {config.seed} | {result.quantization_error:.4f} | {result.occupied_units}/{result.hit_map.size} | {result.dead_units} | {int(result.hit_map.max())} |

El error de cuantización es la distancia euclídea media entre cada muestra y
su BMU, calculada sobre los z-scores; cuanto menor, mejor representa el mapa a
los datos. No alcanza por sí solo para elegir la grilla, porque puede disminuir
simplemente al agregar neuronas. Debe leerse junto con las unidades muertas, la
concentración de hits, la matriz U y los mapas por variable. La comparación con
otras configuraciones se realiza mediante otras corridas, cada una con su
propio JSON y directorio de salida. `followers` no
participa del entrenamiento: su mapa se muestra únicamente para comparar contra
el criterio actual de la agencia.

Los CSV completos están dentro de `generated/`: resumen, estandarización,
asignaciones, matrices y prototipos en unidades originales.

![Mapa de hits](generated/mapa_hits.svg)

![Creadores asignados](generated/creadores.svg)

![Matriz U](generated/matriz_u.svg)

![Mapas por variable](generated/mapas_variables.svg)

![Followers promedio, sólo como baseline](generated/followers.svg)
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run = load_run(args.config)
    dataset = load_creator_dataset(args.data)
    result = run_kohonen(dataset, run)
    output = args.output or Path("docs/kohonen-creadores") / run.name
    output.mkdir(parents=True, exist_ok=True)
    write_tables(output, dataset, result)
    write_plots(output, dataset, result)
    write_report(output, result)
    print(f"Informe Kohonen generado en {output / 'README.md'}")


if __name__ == "__main__":
    main()

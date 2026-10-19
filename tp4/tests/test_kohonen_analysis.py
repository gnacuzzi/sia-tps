"""Pruebas del parser y del runner experimental de Kohonen."""

import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp4.data import CreatorDataset
from sia_tp4.kohonen_analysis import (
    KohonenRun,
    load_run,
    parse_args,
    run_kohonen,
    write_tables,
)
from sia_tp4.kohonen import KohonenConfig
from sia_tp4.kohonen_grid_experiment import (
    GridRunMetrics,
    aggregate_metrics,
    load_grid_runs,
    select_representative_runs,
)


def _raw_run(name="grid-2"):
    return {
        "name": name,
        "grid_size": 2,
        "epochs": 2,
        "learning_rate_start": 0.5,
        "learning_rate_end": 0.1,
        "radius_start": 1.5,
        "radius_end": 1.0,
        "initialization": "sample",
        "shuffle": True,
        "seed": 3,
    }


def _write_config(path, run):
    path.write_text(json.dumps(run), encoding="utf-8")


def test_parser_uses_explicit_config_and_output_paths():
    args = parse_args(
        [
            "--data",
            "input.csv",
            "--config",
            "experiment.json",
            "--output",
            "results",
        ]
    )

    assert str(args.data) == "input.csv"
    assert str(args.config) == "experiment.json"
    assert str(args.output) == "results"


def test_load_run_builds_one_valid_config(tmp_path):
    path = tmp_path / "kohonen.json"
    _write_config(path, _raw_run("first"))

    run = load_run(path)

    assert run.name == "first"
    assert run.config.grid_size == 2
    assert run.config.seed == 3


def test_load_run_rejects_a_list_and_unknown_fields(tmp_path):
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([_raw_run("first")]), encoding="utf-8")
    with pytest.raises(ValueError, match="objeto JSON"):
        load_run(list_path)

    unknown_path = tmp_path / "unknown.json"
    invalid = _raw_run()
    invalid["unsupported"] = 1
    _write_config(unknown_path, invalid)
    with pytest.raises(ValueError, match="sobran"):
        load_run(unknown_path)


def test_runner_writes_numeric_artifacts_for_one_run(tmp_path):
    values = np.asarray(
        [[1.0, 10.0], [2.0, 8.0], [4.0, 3.0], [5.0, 1.0]]
    )
    dataset = CreatorDataset(
        creator_ids=("c1", "c2", "c3", "c4"),
        names=("Uno", "Dos", "Tres", "Cuatro"),
        feature_names=("a", "b"),
        values=values,
        followers=np.asarray([10.0, 20.0, 30.0, 40.0]),
    )
    run = KohonenRun(
        "small",
        KohonenConfig(
            grid_size=2,
            epochs=2,
            learning_rate_start=0.5,
            learning_rate_end=0.1,
            radius_start=1.5,
            radius_end=1.0,
            seed=3,
        ),
    )

    result = run_kohonen(dataset, run)
    write_tables(tmp_path, dataset, result)

    assert result.hit_map.sum() == len(values)
    assert (tmp_path / "generated" / "resumen.csv").is_file()
    assert (tmp_path / "generated" / "mapa_hits.csv").is_file()
    assert (tmp_path / "generated" / "matriz_u.csv").is_file()
    assert (tmp_path / "generated" / "asignaciones.csv").is_file()
    assert (tmp_path / "generated" / "prototipos.csv").is_file()
    assert result.quantization_error >= 0


def test_grid_experiment_has_five_seeds_for_every_size():
    config_directory = (
        Path(__file__).parents[1] / "configs" / "kohonen" / "01-grid"
    )

    runs = load_grid_runs(config_directory)

    assert len(runs) == 35
    for grid_size in range(2, 9):
        group = [run for run in runs if run.config.grid_size == grid_size]
        assert {run.config.seed for run in group} == set(range(5))


def test_grid_metrics_are_aggregated_by_size_with_sample_deviation():
    metrics = [
        GridRunMetrics("a", 2, 0, 1.0, 3, 1, 25.0, 2, 50.0),
        GridRunMetrics("b", 2, 1, 3.0, 1, 3, 75.0, 4, 100.0),
    ]

    aggregate = aggregate_metrics(metrics)

    assert len(aggregate) == 1
    assert aggregate[0]["quantization_error_mean"] == pytest.approx(2.0)
    assert aggregate[0]["quantization_error_std"] == pytest.approx(np.sqrt(2))
    assert aggregate[0]["dead_units_pct_mean"] == pytest.approx(50.0)


def test_representative_run_is_closest_to_median_error():
    runs = [KohonenRun(name, _raw_run_config(seed)) for seed, name in enumerate(("a", "b", "c"))]
    metrics = [
        GridRunMetrics("a", 2, 0, 1.0, 4, 0, 0.0, 2, 50.0),
        GridRunMetrics("b", 2, 1, 3.0, 4, 0, 0.0, 2, 50.0),
        GridRunMetrics("c", 2, 2, 2.0, 4, 0, 0.0, 2, 50.0),
    ]

    selected = select_representative_runs(runs, metrics)

    assert [run.name for run in selected] == ["c"]


def _raw_run_config(seed):
    return KohonenConfig(
        grid_size=2,
        epochs=2,
        learning_rate_start=0.5,
        learning_rate_end=0.1,
        radius_start=1.5,
        radius_end=1.0,
        seed=seed,
    )

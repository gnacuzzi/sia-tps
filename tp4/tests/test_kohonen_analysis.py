"""Pruebas del parser y del runner experimental de Kohonen."""

import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp4.data import BASE_FEATURES, CreatorDataset
from sia_tp4.kohonen_analysis import (
    KohonenRun,
    KohonenRunResult,
    load_run,
    parse_args,
    run_kohonen,
    write_tables,
)
from sia_tp4.kohonen import KohonenConfig, KohonenModel
from sia_tp4.kohonen_grid_experiment import (
    GridRunMetrics,
    aggregate_metrics,
    load_grid_runs,
    select_representative_runs,
)
from sia_tp4.kohonen_rate_experiment import (
    RateRunMetrics,
    aggregate_metrics as aggregate_rate_metrics,
    load_rate_runs,
)
from sia_tp4.kohonen_rate_schedule_experiment import (
    ScheduleRunMetrics,
    aggregate_metrics as aggregate_schedule_metrics,
    load_decay_runs,
    schedule_differences,
    validate_matching_designs,
)
from sia_tp4.kohonen_radius_experiment import (
    RadiusRunMetrics,
    aggregate_metrics as aggregate_radius_metrics,
    load_radius_runs,
    select_representative_runs as select_radius_representative_runs,
)
from sia_tp4.kohonen_epoch_experiment import (
    EpochRunMetrics,
    aggregate_metrics as aggregate_epoch_metrics,
    load_epoch_runs,
)
from sia_tp4.kohonen_initialization_experiment import (
    InitializationRunMetrics,
    initialization_differences,
    load_initialization_runs,
)
from sia_tp4.kohonen_interpretation import (
    InterpretationMetrics,
    dominant_topic_regions,
    load_interpretation_runs,
    select_representatives,
    u_matrix_edge_analysis,
)
from sia_tp4.pca import Standardizer


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


def test_rate_experiment_covers_every_k_rate_and_seed():
    config_directory = (
        Path(__file__).parents[1] / "configs" / "kohonen" / "02-rate"
    )

    runs = load_rate_runs(config_directory)
    combinations = {
        (
            run.config.grid_size,
            run.config.learning_rate_start,
            run.config.seed,
        )
        for run in runs
    }

    assert len(runs) == 245
    assert {combination[0] for combination in combinations} == set(range(2, 9))
    assert {combination[1] for combination in combinations} == {
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        0.7,
        0.9,
    }
    assert {combination[2] for combination in combinations} == set(range(5))


def test_rate_metrics_are_aggregated_by_k_and_rate():
    metrics = [
        RateRunMetrics("a", 4, 0.2, 0, 1.0, 12, 4, 25.0, 3, 6.0),
        RateRunMetrics("b", 4, 0.2, 1, 3.0, 14, 2, 12.5, 4, 8.0),
    ]

    aggregate = aggregate_rate_metrics(metrics)

    assert len(aggregate) == 1
    assert aggregate[0]["grid_size"] == 4
    assert aggregate[0]["learning_rate"] == 0.2
    assert aggregate[0]["quantization_error_mean"] == pytest.approx(2.0)


def test_decay_experiment_matches_constant_design():
    root = Path(__file__).parents[1] / "configs" / "kohonen"
    constant_runs = load_rate_runs(root / "02-rate")
    decay_runs = load_decay_runs(root / "03-rate-decay")

    validate_matching_designs(constant_runs, decay_runs)

    assert len(decay_runs) == 245
    assert all(
        run.config.learning_rate_end
        == pytest.approx(run.config.learning_rate_start * 0.1)
        for run in decay_runs
    )


def test_schedule_difference_subtracts_constant_from_decay():
    metrics = [
        ScheduleRunMetrics(
            "constant-a", "constant", 4, 0.2, 0.2, 0, 2.0, 14, 2, 12.5, 4, 8.0
        ),
        ScheduleRunMetrics(
            "constant-b", "constant", 4, 0.2, 0.2, 1, 4.0, 14, 2, 12.5, 4, 8.0
        ),
        ScheduleRunMetrics(
            "decay-a", "decay", 4, 0.2, 0.02, 0, 1.0, 15, 1, 6.25, 3, 6.0
        ),
        ScheduleRunMetrics(
            "decay-b", "decay", 4, 0.2, 0.02, 1, 3.0, 15, 1, 6.25, 3, 6.0
        ),
    ]

    aggregate = aggregate_schedule_metrics(metrics)
    differences = schedule_differences(aggregate)

    assert differences[0]["quantization_error_delta"] == pytest.approx(-1.0)
    assert differences[0]["dead_units_pct_delta"] == pytest.approx(-6.25)
    assert differences[0]["maximum_hits_pct_delta"] == pytest.approx(-2.0)


def test_radius_experiment_covers_selected_design():
    config_directory = (
        Path(__file__).parents[1] / "configs" / "kohonen" / "04-radius"
    )

    runs = load_radius_runs(config_directory)

    assert len(runs) == 300
    assert {run.config.grid_size for run in runs} == {4, 5, 6}


def test_radius_metrics_include_runtime_in_aggregation():
    metrics = [
        RadiusRunMetrics(
            "a", 4, "constant", 0.1, 0.1, 0.5, 2.0, 1.0, 0,
            0.1, 2.0, 14, 2, 12.5, 4, 8.0, 1.5,
        ),
        RadiusRunMetrics(
            "b", 4, "constant", 0.1, 0.1, 0.5, 2.0, 1.0, 1,
            0.3, 4.0, 14, 2, 12.5, 4, 8.0, 1.7,
        ),
    ]

    aggregate = aggregate_radius_metrics(metrics)

    assert aggregate[0]["training_seconds_mean"] == pytest.approx(0.2)
    assert aggregate[0]["training_seconds_std"] == pytest.approx(np.sqrt(0.02))
    assert aggregate[0]["mean_u_distance_mean"] == pytest.approx(1.6)


def test_radius_representative_is_central_across_selection_metrics():
    runs = []
    metrics = []
    profiles = (
        (4, 0.1, 0.1, 0.0),
        (4, 0.3, 0.03, 0.5),
        (5, 0.2, 0.2, 1.0),
        (5, 0.7, 0.07, 0.5),
        (5, 0.7, 0.07, 0.75),
        (5, 0.9, 0.09, 0.0),
        (6, 0.1, 0.1, 1.0),
        (6, 0.7, 0.07, 1.0),
        (6, 0.9, 0.09, 0.75),
    )
    for profile_index, (grid, rate_start, rate_end, radius_fraction) in enumerate(profiles):
        diameter = np.sqrt(2) * (grid - 1)
        radius_start = 1 + radius_fraction * (diameter - 1)
        for seed in range(5):
            name = f"profile-{profile_index}-seed-{seed}"
            runs.append(
                KohonenRun(
                    name,
                    KohonenConfig(
                        grid_size=grid,
                        epochs=150,
                        learning_rate_start=rate_start,
                        learning_rate_end=rate_end,
                        radius_start=radius_start,
                        radius_end=1.0,
                        seed=seed,
                    ),
                )
            )
            offset = seed
            metrics.append(
                RadiusRunMetrics(
                    name, grid, "constant" if rate_start == rate_end else "decay",
                    rate_start, rate_end, radius_fraction, radius_start, 1.0,
                    seed, 0.1, 2.0 + offset, 10, 2 + offset,
                    10.0 + offset, 4 + offset, 8.0 + offset, 1.5,
                )
            )

    selected = select_radius_representative_runs(runs, metrics)

    assert len(selected) == 9
    assert {run.config.seed for run in selected} == {2}


def test_epoch_experiment_covers_finalists_epochs_and_seeds():
    config_directory = (
        Path(__file__).parents[1] / "configs" / "kohonen" / "05-epochs"
    )

    runs = load_epoch_runs(config_directory)

    assert len(runs) == 150
    assert {run.config.epochs for run in runs} == {25, 50, 100, 150, 250, 400}
    assert {run.config.seed for run in runs} == set(range(5))


def test_epoch_metrics_are_aggregated_by_profile_and_epochs():
    metrics = []
    for seed in range(5):
        metrics.append(
            EpochRunMetrics(
                f"run-{seed}", "profile", 4, 100, 0.1, 0.1, 0.0, 1.0,
                1.0, seed, 0.1 + seed, 2.0 + seed, 14, 2, 12.5, 4,
                8.0, 1.5,
            )
        )

    aggregate = aggregate_epoch_metrics(metrics)

    assert len(aggregate) == 1
    assert aggregate[0]["epochs"] == 100
    assert aggregate[0]["quantization_error_mean"] == pytest.approx(4.0)


def test_initialization_experiment_covers_both_methods_and_five_seeds():
    config_directory = (
        Path(__file__).parents[1]
        / "configs"
        / "kohonen"
        / "06-initialization"
    )

    runs = load_initialization_runs(config_directory)

    assert len(runs) == 50
    assert {run.config.initialization for run in runs} == {"sample", "uniform"}
    assert {run.config.seed for run in runs} == set(range(5))


def test_initialization_differences_are_uniform_minus_sample():
    metrics = []
    profiles = (
        "k4-constant-local",
        "k4-decay-middle",
        "k5-constant-global",
        "k5-decay-wide",
        "k6-decay-global",
    )
    for profile in profiles:
        for initialization, error in (("sample", 3.0), ("uniform", 2.0)):
            for seed in range(5):
                metrics.append(
                    InitializationRunMetrics(
                        f"{profile}-{initialization}-{seed}", profile,
                        initialization, 4, seed, 0.1, error, 14, 2, 12.5,
                        4, 8.0, 1.5,
                    )
                )

    differences = initialization_differences(metrics)

    assert len(differences) == 5
    assert all(
        row["quantization_error_delta_mean"] == pytest.approx(-1.0)
        for row in differences
    )


def test_interpretation_loads_four_sample_profiles_with_five_seeds():
    config_directory = (
        Path(__file__).parents[1]
        / "configs"
        / "kohonen"
        / "06-initialization"
    )

    runs = load_interpretation_runs(config_directory)

    assert len(runs) == 20
    assert {run.config.grid_size for run in runs} == {4, 5}
    assert {run.config.initialization for run in runs} == {"sample"}


def test_interpretation_selects_central_seed_per_profile():
    profiles = (
        "k4-constant-local",
        "k4-decay-middle",
        "k5-constant-global",
        "k5-decay-wide",
    )
    metrics = []
    for profile in profiles:
        for seed in range(5):
            metrics.append(
                InterpretationMetrics(
                    f"{profile}-{seed}",
                    profile,
                    seed,
                    2.0 + seed,
                    5.0 + seed,
                    10.0 + seed,
                )
            )

    selected = select_representatives(metrics)

    assert len(selected) == 4
    assert {metric.seed for metric in selected} == {2}


def test_interpretation_labels_topics_and_compares_neighbor_boundaries():
    feature_indexes = {name: index for index, name in enumerate(BASE_FEATURES)}
    weights = np.zeros((2, 2, len(BASE_FEATURES)), dtype=float)
    topics = (
        (70.0, 10.0, 10.0),
        (10.0, 70.0, 10.0),
        (10.0, 10.0, 70.0),
        (10.0, 10.0, 10.0),
    )
    for (row, column), values in zip(np.ndindex(2, 2), topics):
        for feature, value in zip(
            ("topic_football_pct", "topic_makeup_pct", "topic_dogs_pct"),
            values,
        ):
            weights[row, column, feature_indexes[feature]] = value
    config = KohonenConfig(
        grid_size=2,
        epochs=1,
        learning_rate_start=0.2,
        learning_rate_end=0.2,
        radius_start=1.0,
    )
    standardizer = Standardizer(
        BASE_FEATURES,
        np.zeros(len(BASE_FEATURES)),
        np.ones(len(BASE_FEATURES)),
    )
    model = KohonenModel(weights, BASE_FEATURES, standardizer, config)
    result = KohonenRunResult(
        KohonenRun("topics", config),
        model,
        np.asarray(((0, 0), (0, 1), (1, 0), (1, 1))),
        np.ones((2, 2), dtype=int),
        model.u_matrix(),
        0.0,
    )

    region_map, strengths = dominant_topic_regions(result)
    edges, metrics = u_matrix_edge_analysis(result)

    assert region_map.tolist() == [["football", "makeup"], ["dogs", "art"]]
    assert strengths.tolist() == [[70.0, 70.0], [70.0, 70.0]]
    assert len(edges) == 4
    assert metrics["boundary_edge_count"] == 4

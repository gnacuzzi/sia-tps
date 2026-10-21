"""Pruebas de carga y representación de los QR para Hopfield."""

import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp4.hopfield_qr import (
    QR_FILENAMES,
    QRExperimentDesign,
    _binary_grayscale,
    add_qr_noise,
    edge_sampling_weights,
    energy_trajectory_rows,
    gram_matrices,
    load_design,
    load_qr_patterns,
    pairwise_similarity_rows,
    pixel_balance_rows,
    run_noise_experiment,
    run_zero_noise_recall,
    select_controlled_noise_cases,
    summarize_noise_by_target,
    summarize_noise_runs,
)


DATA_DIRECTORY = Path(__file__).parents[1] / "datos" / "hopfield"


def test_loads_the_five_qr_as_bipolar_matrices_and_vectors():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    assert patterns.names == (
        "beverages",
        "dessert",
        "food",
        "specials",
        "wine",
    )
    assert patterns.images.shape == (5, 37, 37)
    assert patterns.vectors.shape == (5, 37 * 37)
    assert patterns.images.dtype == np.int8
    assert set(np.unique(patterns.images)) == {-1, 1}
    np.testing.assert_array_equal(
        patterns.vectors,
        patterns.images.reshape(5, -1),
    )


def test_black_is_positive_and_white_is_negative():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    # En el primer QR, la esquina pertenece al patrón de búsqueda negro y el
    # módulo siguiente a ese patrón es blanco.
    assert patterns.images[0, 0, 0] == 1
    assert patterns.images[0, 0, 7] == -1


def test_binary_validation_rejects_wrong_shape_channels_and_gray_values():
    path = Path("invalid.png")

    with pytest.raises(ValueError, match="37x37"):
        _binary_grayscale(np.zeros((5, 5)), path)
    with pytest.raises(ValueError, match="escala de grises"):
        _binary_grayscale(np.zeros((37, 37, 3)), path)
    with pytest.raises(ValueError, match="debe ser binaria"):
        _binary_grayscale(np.full((37, 37), 0.5), path)


def test_loader_reports_a_missing_expected_file(tmp_path):
    with pytest.raises(ValueError, match=QR_FILENAMES[0]):
        load_qr_patterns(tmp_path)


def test_gram_matrix_is_symmetric_normalized_and_does_not_overflow():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    gram, normalized = gram_matrices(patterns)

    assert gram.dtype == np.int64
    np.testing.assert_array_equal(gram, gram.T)
    np.testing.assert_array_equal(np.diag(gram), 37 * 37)
    np.testing.assert_allclose(normalized, gram / (37 * 37))
    np.testing.assert_allclose(np.diag(normalized), 1.0)


def test_pairwise_rows_cover_every_pair_and_match_bipolar_distance():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    rows = pairwise_similarity_rows(patterns)

    assert len(rows) == 10
    for row in rows:
        assert (
            row["inner_product"]
            == 37 * 37 - 2 * row["different_pixels"]
        )
    closest = max(rows, key=lambda row: row["absolute_inner_product"])
    assert (closest["first"], closest["second"]) == ("food", "wine")
    assert closest["inner_product"] == 1177


def test_pixel_balance_accounts_for_every_pixel():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    rows = pixel_balance_rows(patterns)

    assert len(rows) == 5
    assert all(
        row["black_pixels"] + row["white_pixels"] == 37 * 37
        for row in rows
    )


def test_five_stored_qr_collapse_to_one_spurious_state_without_noise():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    rows, cases = run_zero_noise_recall(patterns)

    assert len(rows) == len(cases) == 5
    assert all(not row["remains_stable"] for row in rows)
    assert all(row["outcome"] == "spurious_fixed" for row in rows)
    assert all(row["status"] == "fixed_point" for row in rows)
    assert all(row["matched_pattern"] == "" for row in rows)
    assert [row["iterations"] for row in rows] == [2, 2, 2, 2, 2]
    assert [row["final_error_pixels"] for row in rows] == [225, 225, 71, 103, 91]
    assert len({case["result"].final_state.tobytes() for case in cases}) == 1
    for case in cases:
        result = case["result"]
        assert len(result.energies) == result.iterations + 1
        assert np.all(np.diff(result.energies) <= 1e-9)

    energy_rows = energy_trajectory_rows(cases)
    assert len(energy_rows) == 5 * 3
    assert [row["iteration"] for row in energy_rows[:3]] == [0, 1, 2]
    assert all(
        row["error_pixels"] == 0
        for row in energy_rows
        if row["iteration"] == 0
    )


def test_qr_design_defines_three_noises_and_at_least_ten_seeds():
    path = Path(__file__).parents[1] / "configs" / "hopfield" / "qr.json"

    design = load_design(path)

    assert design.noise_types == (
        "uniform_flip",
        "uniform_erasure",
        "edge_erasure",
    )
    assert design.noise_levels[0] == 0.0
    assert len(design.seeds) >= 10


def test_qr_design_rejects_fewer_than_ten_seeds(tmp_path):
    path = tmp_path / "qr.json"
    path.write_text(
        json.dumps(
            {
                "noise_types": ["uniform_flip"],
                "noise_levels": [0.0, 0.1],
                "seeds": list(range(9)),
                "max_iterations": 20,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="al menos 10"):
        load_design(path)


def test_noise_functions_are_exact_directional_and_reproducible():
    target = load_qr_patterns(DATA_DIRECTORY).vectors[0]

    flipped, selected_flip, changed_flip = add_qr_noise(
        target, "uniform_flip", 0.1, seed=3
    )
    repeated, _, _ = add_qr_noise(
        target, "uniform_flip", 0.1, seed=3
    )
    erased, selected_erasure, changed_erasure = add_qr_noise(
        target, "uniform_erasure", 0.1, seed=3
    )

    assert selected_flip == selected_erasure == round(0.1 * 37 * 37)
    assert changed_flip == selected_flip
    assert changed_erasure <= selected_erasure
    np.testing.assert_array_equal(flipped, repeated)
    assert np.all(erased[target == -1] == -1)
    assert not np.any((target == -1) & (erased == 1))


def test_edge_weights_are_symmetric_and_larger_at_the_border():
    weights = edge_sampling_weights().reshape(37, 37)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0, 0] > weights[18, 18]
    np.testing.assert_allclose(weights, weights[::-1, :])
    np.testing.assert_allclose(weights, weights[:, ::-1])


def test_noise_experiment_uses_ten_seeds_and_selects_cases_without_outcomes():
    patterns = load_qr_patterns(DATA_DIRECTORY)
    design = QRExperimentDesign(
        noise_types=("uniform_flip",),
        noise_levels=(0.0, 0.1),
        seeds=tuple(range(10)),
        max_iterations=20,
    )
    zero_rows, _ = run_zero_noise_recall(patterns, design.max_iterations)

    rows, cases = run_noise_experiment(patterns, design)
    summary = summarize_noise_runs(rows, zero_rows, design)
    target_summary = summarize_noise_by_target(rows, zero_rows, design)
    selected = select_controlled_noise_cases(cases, design, patterns.names[0])

    assert len(rows) == 5 * 10
    assert len(cases) == len(rows)
    assert all(
        row["attractor"]
        in {
            "baseline_spurious",
            "inverse_baseline_spurious",
            "other_spurious",
            "cycle",
            "max_iterations",
            *(f"stored_{name}" for name in patterns.names),
        }
        for row in rows
    )
    assert len(summary) == 2
    assert summary[0]["seed_count"] == 0
    assert summary[1]["seed_count"] == 10
    assert len(target_summary) == 5 * 2
    assert all(
        row["seed_count"] == 10
        for row in target_summary
        if row["noise_level"] > 0
    )
    assert len(selected) == 1
    assert selected[0]["row"]["target"] == patterns.names[0]
    assert selected[0]["row"]["noise_level"] == 0.1
    assert selected[0]["row"]["seed"] == 0

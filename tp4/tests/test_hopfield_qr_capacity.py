"""Pruebas del barrido exhaustivo de capacidad con QR."""

import json
from pathlib import Path

import pytest

from sia_tp4.hopfield_qr import load_qr_patterns
from sia_tp4.hopfield_qr_capacity import (
    load_capacity_design,
    qr_subsets,
    run_capacity_sweep,
    summarize_capacity_by_size,
)


DATA_DIRECTORY = Path(__file__).parents[1] / "datos" / "hopfield"


def test_capacity_design_is_minimal_and_validated(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text('{"max_iterations": 50}', encoding="utf-8")
    assert load_capacity_design(valid).max_iterations == 50

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"max_iterations": 50, "seeds": list(range(10))}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="desconocidos"):
        load_capacity_design(invalid)


def test_all_nonempty_subsets_are_enumerated_by_size():
    subsets = qr_subsets(5)

    assert len(subsets) == 31
    assert [sum(len(item) == size for item in subsets) for size in range(1, 6)] == [
        5,
        10,
        10,
        5,
        1,
    ]
    assert len(set(subsets)) == len(subsets)


def test_capacity_sweep_covers_80_queries_and_finds_the_observed_cutoff():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    query_rows, subset_rows = run_capacity_sweep(patterns)
    size_rows = summarize_capacity_by_size(subset_rows)

    assert len(query_rows) == 80
    assert len(subset_rows) == 31
    assert [row["query_count"] for row in size_rows] == [5, 20, 30, 20, 5]
    assert [row["fully_stable_subset_count"] for row in size_rows] == [
        5,
        10,
        0,
        0,
        0,
    ]
    assert [row["stable_query_count"] for row in size_rows] == [
        5,
        20,
        0,
        0,
        0,
    ]
    assert all(row["all_energy_non_increasing"] for row in subset_rows)
    assert all(
        row["pair_count"] == 0
        and row["mean_absolute_normalized_inner_product"] == ""
        for row in subset_rows
        if row["subset_size"] == 1
    )


def test_most_correlated_pair_is_still_fully_stable():
    patterns = load_qr_patterns(DATA_DIRECTORY)
    _, subset_rows = run_capacity_sweep(patterns)
    pairs = [row for row in subset_rows if row["subset_size"] == 2]

    most_correlated = max(
        pairs,
        key=lambda row: row["max_absolute_normalized_inner_product"],
    )
    assert most_correlated["stored_patterns"] == "food+wine"
    assert most_correlated["fully_stable"] is True

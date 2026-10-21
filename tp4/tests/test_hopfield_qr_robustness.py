"""Pruebas del barrido de robustez con uno y dos QR."""

from pathlib import Path

from sia_tp4.hopfield_qr import (
    QRExperimentDesign,
    load_qr_patterns,
)
from sia_tp4.hopfield_qr_robustness import (
    robustness_subsets,
    run_robustness_sweep,
    summarize_by_storage_size,
    summarize_by_subset,
    summarize_pair_robustness,
)


DATA_DIRECTORY = Path(__file__).parents[1] / "datos" / "hopfield"


def _small_design():
    return QRExperimentDesign(
        noise_types=("uniform_flip",),
        noise_levels=(0.0, 0.1),
        seeds=tuple(range(10)),
        max_iterations=20,
    )


def test_robustness_subsets_cover_five_individuals_and_ten_pairs():
    subsets = robustness_subsets(5)

    assert len(subsets) == 15
    assert sum(len(subset) == 1 for subset in subsets) == 5
    assert sum(len(subset) == 2 for subset in subsets) == 10


def test_small_sweep_is_complete_paired_and_starts_from_stable_memories():
    patterns = load_qr_patterns(DATA_DIRECTORY)
    design = _small_design()

    baseline, noisy = run_robustness_sweep(patterns, design)

    assert len(baseline) == 25
    assert len(noisy) == 25 * 10
    assert all(row["remains_stable"] for row in baseline)
    beverages = [
        row
        for row in noisy
        if row["target"] == "beverages" and row["seed"] == 3
    ]
    assert len(beverages) == 5
    assert len({row["changed_pixels"] for row in beverages}) == 1


def test_summaries_preserve_sizes_subsets_levels_and_ten_seeds():
    patterns = load_qr_patterns(DATA_DIRECTORY)
    design = _small_design()
    baseline, noisy = run_robustness_sweep(patterns, design)

    by_size = summarize_by_storage_size(baseline, noisy, design)
    by_subset = summarize_by_subset(baseline, noisy, design)
    pairs = summarize_pair_robustness(by_subset)

    assert len(by_size) == 2 * 2
    assert len(by_subset) == 15 * 2
    assert all(row["seed_count"] == 10 for row in by_size if row["noise_level"] > 0)
    assert all(row["seed_count"] == 10 for row in by_subset if row["noise_level"] > 0)
    assert len(pairs) == 10
    assert all(row["recovery_at_10_percent"] >= 0 for row in pairs)

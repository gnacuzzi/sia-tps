import numpy as np

from sia_tp5.denoising_final_analysis import (aggregate_curves,
                                              select_representative_cases)


def _summary(probability, output_error, exact, worsened):
    return {
        "noise_probability": probability,
        "mean_input_pixel_errors": 2.0,
        "mean_output_pixel_errors": output_error,
        "exact_recovery_rate": exact,
        "worsened_rate": worsened,
    }


def test_aggregate_curves_uses_variation_between_seeds():
    rows = aggregate_curves([
        [_summary(0.1, 1.0, 0.8, 0.1)],
        [_summary(0.1, 3.0, 0.6, 0.2)],
    ])

    assert len(rows) == 1
    assert rows[0]["seeds"] == 2
    assert rows[0]["mean_output_pixel_errors_mean"] == 2.0
    assert rows[0]["mean_output_pixel_errors_std"] == 1.0
    assert np.isclose(rows[0]["exact_recovery_rate_mean"], 0.7)


def _case(probability, before, after, seed=0, repetition=0, index=0):
    pixels = np.zeros(35)
    return {
        "seed": seed,
        "noise_probability": probability,
        "repetition": repetition,
        "label": "a",
        "index": index,
        "input_pixel_errors": before,
        "output_pixel_errors": after,
        "improvement": before - after,
        "clean": pixels,
        "noisy": pixels,
        "output": pixels,
    }


def test_case_selection_respects_the_three_fixed_definitions():
    cases = [
        _case(0.1, 3, 0, index=0),
        _case(0.1, 7, 0, index=1),
        _case(0.2, 6, 2, index=2),
        _case(0.2, 8, 3, index=3),
        _case(0.3, 5, 8, index=4),
        _case(0.3, 7, 11, index=5),
    ]

    selected = select_representative_cases(cases)

    assert selected["easy"]["input_pixel_errors"] == 7
    assert 0 < selected["difficult"]["output_pixel_errors"] \
        < selected["difficult"]["input_pixel_errors"]
    assert selected["failed"]["output_pixel_errors"] \
        > selected["failed"]["input_pixel_errors"]

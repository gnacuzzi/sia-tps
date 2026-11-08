import csv
import json

from sia_tp5.calibration import planned_trials, run_calibration


def test_calibration_expands_rates_and_architectures_without_figures(tmp_path):
    configuration = {
        "name": "calibration-smoke-test",
        "architectures": {
            "small": {
                "layers": [35, 3, 2, 3, 35],
                "latent_layer": 2,
                "activations": ["tanh", "linear", "tanh", "logistic"],
            },
        },
        "optimizers": [
            {
                "name": "gradient_descent",
                "learning_rates": [0.001, 0.01],
            },
        ],
        "seeds": [0],
        "max_epochs": 2,
        "requirement_max_pixel_errors": 1,
    }

    output, summaries = run_calibration(configuration, tmp_path)

    assert len(summaries) == 2
    assert {row["learning_rate"] for row in summaries} == {0.001, 0.01}
    assert all(row["epochs_completed"] == 2 for row in summaries)
    assert all(row["status"] == "completed" for row in summaries)
    assert all(row["minimum_loss_epoch"] <= 2 for row in summaries)
    assert all(row["final_minus_minimum_loss"] >= 0 for row in summaries)
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*.npz"))
    assert len(list((output / "histories").glob("*.csv"))) == 2
    assert json.loads((output / "spec.json").read_text()) == configuration
    with (output / "summary.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 2


def test_dry_plan_has_no_side_effects(tmp_path):
    configuration = {
        "architectures": {"a": {}, "b": {}},
        "optimizers": [
            {"name": "adam", "learning_rates": [0.001, 0.01]},
        ],
        "seeds": [0, 1],
        "max_epochs": 10,
    }

    trials = planned_trials(configuration)

    assert len(trials) == 8
    assert len({trial["run_id"] for trial in trials}) == 8
    assert not list(tmp_path.iterdir())


def test_explicit_combinations_do_not_create_a_cartesian_product():
    configuration = {
        "architectures": {"a": {}, "b": {}},
        "combinations": [
            {
                "architecture": "a",
                "optimizer": {
                    "name": "gradient_descent",
                    "learning_rate": 0.1,
                },
            },
            {
                "architecture": "b",
                "optimizer": {
                    "name": "adam",
                    "learning_rate": 0.01,
                },
            },
        ],
        "seeds": [1, 2],
        "max_epochs": 10,
    }

    trials = planned_trials(configuration)

    assert len(trials) == 4
    assert {(trial["architecture"], trial["optimizer"])
            for trial in trials} == {("a", "gradient_descent"), ("b", "adam")}

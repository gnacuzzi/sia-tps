"""Elegir casos generados e interpolaciones de codigo mediante reglas fijas."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .code_dataset import (belongs_to_grammar, compiles, decode_patterns,
                           load_code_dataset)
from .variational_autoencoder import VariationalAutoencoder


def _classify(source, train_sources):
    compilable = compiles(source)
    grammar = belongs_to_grammar(source)
    return {
        "compiles": compilable,
        "belongs_to_grammar": grammar,
        "novel_vs_train": grammar and source not in train_sources,
    }


def analyze(configuration):
    dataset = load_code_dataset(
        split_seed=configuration["split_seed"],
        train_fraction=configuration["train_fraction"],
    )
    train_sources = set(dataset.train_sources)
    model = VariationalAutoencoder.load(configuration["model_path"])
    latent_size = model.architecture[model.latent_layer]
    latent = np.random.default_rng(configuration["prior_seed"]).standard_normal(
        (configuration["prior_sample_count"], latent_size)
    )
    generated = decode_patterns(model.decode(latent))
    records = [
        {
            "sample": index,
            "source": source,
            **_classify(source, train_sources),
        }
        for index, source in enumerate(generated)
    ]
    rules = {
        "valid_novel": lambda row: row["novel_vs_train"],
        "valid_seen": lambda row: (
            row["belongs_to_grammar"] and not row["novel_vs_train"]
        ),
        "compilable_outside_grammar": lambda row: (
            row["compiles"] and not row["belongs_to_grammar"]
        ),
        "invalid_syntax": lambda row: not row["compiles"],
    }
    representatives = {}
    for name, rule in rules.items():
        representatives[name] = next(
            (row for row in records if rule(row)), None
        )

    test = dataset.test_patterns.astype(np.int64)
    distances = np.sum(test[:, None, :] != test[None, :, :], axis=2)
    distances[np.tril_indices(len(test))] = -1
    left_index, right_index = np.argwhere(
        distances == distances.max()
    )[0]
    endpoint_patterns = dataset.test_patterns[[left_index, right_index]]
    endpoint_mu, _ = model.encode_distribution(endpoint_patterns)
    t_values = np.linspace(0.0, 1.0, configuration["interpolation_points"])
    interpolation_latent = (
        (1.0 - t_values[:, None]) * endpoint_mu[0]
        + t_values[:, None] * endpoint_mu[1]
    )
    interpolation_sources = decode_patterns(model.decode(interpolation_latent))
    interpolation = [
        {
            "t": float(t),
            "source": source,
            **_classify(source, train_sources),
        }
        for t, source in zip(t_values, interpolation_sources)
    ]
    payload = {
        "representative_selection": "primer indice que satisface cada regla",
        "representatives": representatives,
        "interpolation_selection": (
            "primer par de test con maxima distancia de Hamming"
        ),
        "interpolation_hamming_distance": int(distances.max()),
        "left_source": dataset.test_sources[left_index],
        "right_source": dataset.test_sources[right_index],
        "interpolation": interpolation,
    }
    output = Path(configuration["analysis_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    prior_output = Path(configuration["prior_output"])
    prior_output.parent.mkdir(parents=True, exist_ok=True)
    with prior_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "sample", "source_json", "compiles", "belongs_to_grammar",
            "novel_vs_train",
        ])
        writer.writeheader()
        for row in records:
            writer.writerow({
                "sample": row["sample"],
                "source_json": json.dumps(row["source"]),
                "compiles": row["compiles"],
                "belongs_to_grammar": row["belongs_to_grammar"],
                "novel_vs_train": row["novel_vs_train"],
            })
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    payload = analyze(configuration)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Construir y registrar el split deduplicado del ejercicio 3."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sia_tp3.digit_development import load_unique_digit_development
from sia_tp3.experiments import _temporary_validation_indices


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(primary_path, additional_path, output, *, validation_fraction=0.2,
        validation_seed=0):
    primary_path = Path(primary_path)
    additional_path = Path(additional_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("usar una carpeta de salida nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)

    development = load_unique_digit_development(primary_path, additional_path)
    train_indices, validation_indices = _temporary_validation_indices(
        development.y, validation_fraction, validation_seed)
    train_set = set(int(index) for index in train_indices)
    validation_set = set(int(index) for index in validation_indices)
    if train_set & validation_set:
        raise RuntimeError("training y validation no son disjuntos")
    if train_set | validation_set != set(range(len(development.y))):
        raise RuntimeError("el split no cubre todo development")

    np.savez_compressed(
        output / "split-indices.npz",
        train=train_indices,
        validation=validation_indices,
    )
    manifest = []
    for index, (image, label, primary_row, additional_row) in enumerate(zip(
            development.X, development.y, development.primary_rows,
            development.additional_rows)):
        manifest.append({
            "canonical_index": index,
            "label": int(label),
            "partition": "training" if index in train_set else "validation",
            "digits_csv_row": "" if primary_row < 0 else int(primary_row),
            "more_digits_csv_row": (
                "" if additional_row < 0 else int(additional_row)),
            "present_in_both_sources": bool(
                primary_row >= 0 and additional_row >= 0),
            "image_sha256": hashlib.sha256(
                np.ascontiguousarray(image).tobytes()).hexdigest(),
        })
    write_csv(output / "manifest.csv", manifest)

    class_rows = []
    for label in range(10):
        total = int((development.y == label).sum())
        training = int((development.y[train_indices] == label).sum())
        validation = int((development.y[validation_indices] == label).sum())
        class_rows.append({
            "class": label,
            "total": total,
            "training": training,
            "validation": validation,
            "validation_fraction": validation / total,
        })
    write_csv(output / "class-balance.csv", class_rows)

    summary = {
        "primary_path": str(primary_path),
        "primary_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
        "additional_path": str(additional_path),
        "additional_sha256": hashlib.sha256(
            additional_path.read_bytes()).hexdigest(),
        "test_opened": False,
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
        "rows_before_deduplication": (
            len(development.y) + development.extra_copies),
        "extra_copies_removed": development.extra_copies,
        "unique_development_rows": len(development.y),
        "training_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "conflicting_labels": 0,
        "training_validation_overlap": 0,
        "classes": class_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=0)
    args = parser.parse_args()
    result = run(
        args.primary,
        args.additional,
        args.output,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

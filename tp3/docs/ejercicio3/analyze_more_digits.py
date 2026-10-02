"""EDA reproducible de more_digits.csv y su relación con digits.csv.

No entrena modelos. De digits_test.csv sólo lee los píxeles para comprobar
solapamientos exactos; sus etiquetas no se interpretan.
"""

import argparse
import ast
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3.data import _load_digit_file


LABELS = tuple(range(10))
SOURCES = ("digits.csv", "more_digits.csv")


def write_csv(path, rows):
    if not rows:
        raise ValueError("no se pueden guardar filas vacías")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_inputs(path):
    """Leer imágenes de test sin convertir ni devolver la columna label."""
    images = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != {"label", "image"}:
            raise ValueError("columnas inesperadas en digits_test.csv")
        for row_number, row in enumerate(reader, start=2):
            try:
                image = np.asarray(ast.literal_eval(row["image"]), dtype=np.float32)
            except (SyntaxError, TypeError, ValueError) as error:
                raise ValueError(
                    f"imagen de test inválida en la fila {row_number}") from error
            if image.shape != (784,) or not np.isfinite(image).all():
                raise ValueError(f"imagen de test inválida en la fila {row_number}")
            images.append(image)
    if not images:
        raise ValueError("digits_test.csv está vacío")
    return np.stack(images)


def row_keys(images):
    contiguous = np.ascontiguousarray(images, dtype=np.float32)
    return [row.tobytes() for row in contiguous]


def source_quality(name, images, labels):
    counts = np.bincount(labels, minlength=10)
    unique_values = np.unique(images)
    return {
        "dataset": name,
        "rows": int(len(images)),
        "pixels_per_image": int(images.shape[1]),
        "dtype": str(images.dtype),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
        "mean": float(images.mean(dtype=np.float64)),
        "std": float(images.std(dtype=np.float64)),
        "zero_fraction": float((images == 0).mean()),
        "unique_pixel_values": int(len(unique_values)),
        "blank_images": int(np.all(images == 0, axis=1).sum()),
        "constant_images": int((np.ptp(images, axis=1) == 0).sum()),
        "constant_pixels": int((np.ptp(images, axis=0) == 0).sum()),
        "absent_classes": [int(label) for label in LABELS if counts[label] == 0],
    }


def internal_duplicates(images, labels):
    groups = defaultdict(list)
    for index, (key, label) in enumerate(zip(row_keys(images), labels)):
        groups[key].append((index, int(label)))
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    return {
        "duplicate_input_groups": len(repeated),
        "duplicate_input_rows": sum(len(rows) for rows in repeated),
        "extra_input_copies": sum(len(rows) - 1 for rows in repeated),
        "conflicting_label_groups": sum(
            len({label for _, label in rows}) > 1 for rows in repeated),
    }


def compare_sources(old_images, old_labels, new_images, new_labels):
    """Caracterizar imágenes compartidas sin asumir una correspondencia 1 a 1."""
    groups = defaultdict(lambda: {"old": [], "new": []})
    for index, (key, label) in enumerate(zip(row_keys(old_images), old_labels)):
        groups[key]["old"].append((index, int(label)))
    for index, (key, label) in enumerate(zip(row_keys(new_images), new_labels)):
        groups[key]["new"].append((index, int(label)))

    shared = [group for group in groups.values() if group["old"] and group["new"]]
    conflicts = []
    for group in shared:
        old_set = {label for _, label in group["old"]}
        new_set = {label for _, label in group["new"]}
        if old_set != new_set or len(old_set | new_set) > 1:
            conflicts.append(group)

    return {
        "shared_input_groups": len(shared),
        "old_rows_in_shared_groups": sum(len(group["old"]) for group in shared),
        "new_rows_in_shared_groups": sum(len(group["new"]) for group in shared),
        "shared_groups_with_label_conflict": len(conflicts),
        "unique_input_groups_in_union": len(groups),
        "union_rows_before_deduplication": len(old_images) + len(new_images),
        "extra_input_copies_in_union": (
            len(old_images) + len(new_images) - len(groups)),
    }, groups


def overlap_with_test(source_name, images, test_images):
    source_keys = row_keys(images)
    test_keys = row_keys(test_images)
    shared = set(source_keys) & set(test_keys)
    return {
        "dataset": source_name,
        "shared_input_groups": len(shared),
        "development_rows_in_shared_groups": sum(key in shared for key in source_keys),
        "test_rows_in_shared_groups": sum(key in shared for key in test_keys),
    }


def class_rows(name, labels):
    return [{
        "dataset": name,
        "class": label,
        "count": int((labels == label).sum()),
        "fraction": float((labels == label).mean()),
    } for label in LABELS]


def class_mean_comparison(old_images, old_labels, new_images, new_labels):
    rows = []
    for label in LABELS:
        old = old_images[old_labels == label]
        new = new_images[new_labels == label]
        if len(old) and len(new):
            difference = old.mean(axis=0, dtype=np.float64) - new.mean(
                axis=0, dtype=np.float64)
            rmse = float(np.sqrt(np.mean(difference ** 2)))
            mean_absolute_difference = float(np.mean(np.abs(difference)))
        else:
            rmse = ""
            mean_absolute_difference = ""
        rows.append({
            "class": label,
            "digits_count": len(old),
            "more_digits_count": len(new),
            "mean_image_rmse": rmse,
            "mean_image_mean_absolute_difference": mean_absolute_difference,
        })
    return rows


def union_class_balance(groups):
    """Separar por clase lo compartido y el aporte único de cada archivo."""
    counts = {label: {"only_digits": 0, "shared": 0, "only_more_digits": 0}
              for label in LABELS}
    for group in groups.values():
        labels = {label for _, label in group["old"] + group["new"]}
        if len(labels) != 1:
            continue
        label = labels.pop()
        if group["old"] and group["new"]:
            counts[label]["shared"] += 1
        elif group["old"]:
            counts[label]["only_digits"] += 1
        else:
            counts[label]["only_more_digits"] += 1
    total = sum(sum(values.values()) for values in counts.values())
    return [{
        "class": label,
        **counts[label],
        "unique_union_count": sum(counts[label].values()),
        "unique_union_fraction": sum(counts[label].values()) / total,
    } for label in LABELS]


def save_figures(output, old_images, old_labels, new_images, new_labels, seed):
    colors = {"digits.csv": "#2878a0", "more_digits.csv": "#d97925"}
    old_counts = np.bincount(old_labels, minlength=10)
    new_counts = np.bincount(new_labels, minlength=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(10)
    width = 0.38
    axes[0].bar(x - width / 2, old_counts, width, label="digits.csv",
                color=colors["digits.csv"])
    axes[0].bar(x + width / 2, new_counts, width, label="more_digits.csv",
                color=colors["more_digits.csv"])
    axes[0].set(title="Cantidad por clase", xlabel="Dígito", ylabel="Imágenes",
                xticks=x)
    axes[0].legend()
    axes[1].bar(x - width / 2, old_counts / len(old_labels), width,
                label="digits.csv", color=colors["digits.csv"])
    axes[1].bar(x + width / 2, new_counts / len(new_labels), width,
                label="more_digits.csv", color=colors["more_digits.csv"])
    axes[1].set(title="Proporción por clase", xlabel="Dígito", ylabel="Fracción",
                xticks=x)
    axes[1].legend()
    fig.suptitle("Balance de clases por archivo")
    fig.savefig(output / "class-balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 10, figsize=(15, 5), constrained_layout=True)
    for label in LABELS:
        old = old_images[old_labels == label]
        new = new_images[new_labels == label]
        entries = [
            (old.mean(axis=0) if len(old) else None, "digits"),
            (new.mean(axis=0) if len(new) else None, "more"),
        ]
        for row, (image, source) in enumerate(entries):
            ax = axes[row, label]
            ax.axis("off")
            if image is None:
                ax.text(0.5, 0.5, "sin datos", ha="center", va="center")
            else:
                ax.imshow(image.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{source} · {label}", fontsize=8)
        ax = axes[2, label]
        ax.axis("off")
        if len(old) and len(new):
            difference = old.mean(axis=0) - new.mean(axis=0)
            ax.imshow(difference.reshape(28, 28), cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        else:
            ax.text(0.5, 0.5, "no comparable", ha="center", va="center")
        ax.set_title(f"diferencia · {label}", fontsize=8)
    fig.suptitle("Imagen media por clase y diferencia digits - more_digits")
    fig.savefig(output / "class-mean-images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 10, figsize=(14, 5), constrained_layout=True)
    example_rows = []
    for label in LABELS:
        available = np.flatnonzero(new_labels == label)
        chosen = rng.choice(available, size=min(3, len(available)), replace=False)
        for row in range(3):
            ax = axes[row, label]
            ax.axis("off")
            if row < len(chosen):
                index = int(chosen[row])
                ax.imshow(new_images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{label} · fila {index + 2}", fontsize=8)
                example_rows.append({"class": label, "csv_row": index + 2})
    fig.suptitle(f"Ejemplos de more_digits.csv · selección con semilla {seed}")
    fig.savefig(output / "more-digits-examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_csv(output / "example-rows.csv", example_rows)


def run(data_dir, output, seed=0):
    data_dir, output = Path(data_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("usar una carpeta de salida nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print("Cargando digits.csv...", flush=True)
    old_images, old_labels = _load_digit_file(data_dir / "digits.csv")
    print("Cargando more_digits.csv...", flush=True)
    new_images, new_labels = _load_digit_file(data_dir / "more_digits.csv")
    print("Leyendo únicamente los píxeles de digits_test.csv...", flush=True)
    external_test_images = test_inputs(data_dir / "digits_test.csv")

    qualities = [
        source_quality("digits.csv", old_images, old_labels),
        source_quality("more_digits.csv", new_images, new_labels),
    ]
    balances = class_rows("digits.csv", old_labels) + class_rows(
        "more_digits.csv", new_labels)
    internal = {
        "digits.csv": internal_duplicates(old_images, old_labels),
        "more_digits.csv": internal_duplicates(new_images, new_labels),
    }
    between, groups = compare_sources(
        old_images, old_labels, new_images, new_labels)
    overlaps = [
        overlap_with_test("digits.csv", old_images, external_test_images),
        overlap_with_test("more_digits.csv", new_images, external_test_images),
        overlap_with_test(
            "union_unique_inputs",
            np.stack([np.frombuffer(key, dtype=np.float32) for key in groups]),
            external_test_images,
        ),
    ]
    mean_comparison = class_mean_comparison(
        old_images, old_labels, new_images, new_labels)
    union_balance = union_class_balance(groups)

    write_csv(output / "dataset-quality.csv", qualities)
    write_csv(output / "class-balance.csv", balances)
    write_csv(output / "class-mean-comparison.csv", mean_comparison)
    write_csv(output / "union-class-balance.csv", union_balance)
    write_csv(output / "test-input-overlap.csv", overlaps)
    duplicate_columns = (
        "duplicate_input_groups", "duplicate_input_rows", "extra_input_copies",
        "conflicting_label_groups", "shared_input_groups",
        "old_rows_in_shared_groups", "new_rows_in_shared_groups",
        "shared_groups_with_label_conflict", "unique_input_groups_in_union",
        "union_rows_before_deduplication", "extra_input_copies_in_union",
    )
    duplicate_rows = []
    for comparison, values in (
            ("within_digits", internal["digits.csv"]),
            ("within_more_digits", internal["more_digits.csv"]),
            ("between_sources", between)):
        duplicate_rows.append({
            "comparison": comparison,
            **{column: values.get(column, "") for column in duplicate_columns},
        })
    write_csv(output / "duplicate-summary.csv", duplicate_rows)
    save_figures(output, old_images, old_labels, new_images, new_labels, seed)

    metadata = {
        "seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "sha256": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("digits.csv", "more_digits.csv", "digits_test.csv")
        },
        "test_labels_read": False,
    }
    result = {
        "metadata": metadata,
        "quality": qualities,
        "class_balance": balances,
        "internal_duplicates": internal,
        "between_sources": between,
        "test_input_overlap": overlaps,
        "class_mean_comparison": mean_comparison,
        "union_class_balance": union_balance,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Resultados guardados en {output}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    run(arguments.data, arguments.output, arguments.seed)

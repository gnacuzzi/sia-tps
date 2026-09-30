"""EDA reproducible de los datasets; conserva los datos y no evalúa modelos."""

import argparse
import ast
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3.data import FRAUD_FEATURES, FRAUD_LABEL, FRAUD_TARGET, _load_digit_file


UNITS = ["segundos Unix", "USD", "unidades", "segundos", "días", "días",
         "píxeles (ancho × alto)", "segundos", "unidades", "probabilidad"]
INTEGER_FEATURES = {"timestamp", "quantity_purchased", "account_age_days",
                    "device_screen_resolution", "items_viewed_before_purchase"}


def write_csv(path, rows, *, lineterminator="\r\n"):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]),
                                lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)


def describe(name, values, unit):
    """Calcular estadísticos sobre valores finitos, informando los excluidos."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = dict(variable=name, unit=unit, count=len(values),
                  missing_or_nan=int(np.isnan(values).sum()),
                  infinite=int(np.isinf(values).sum()), finite_count=len(finite),
                  unique=int(len(np.unique(finite))))
    for key in ("min", "p01", "q1", "median", "q3", "p99", "max", "mean",
                "std", "iqr", "lower_fence", "upper_fence", "iqr_candidates"):
        result[key] = None
    if len(finite):
        quantiles = np.percentile(finite, [0, 1, 25, 50, 75, 99, 100])
        result.update(zip(("min", "p01", "q1", "median", "q3", "p99", "max"),
                          map(float, quantiles)))
        iqr = result["q3"] - result["q1"]
        low, high = result["q1"] - 1.5 * iqr, result["q3"] + 1.5 * iqr
        result.update(mean=float(finite.mean()), std=float(finite.std(ddof=0)),
                      iqr=iqr, lower_fence=low, upper_fence=high,
                      iqr_candidates=int(((finite < low) | (finite > high)).sum()))
    return result


def fraud_data(path):
    """Leer entradas, objetivo y etiqueta de todas las transacciones."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = set(FRAUD_FEATURES) | {FRAUD_TARGET, FRAUD_LABEL}
        if set(reader.fieldnames or []) != expected:
            raise ValueError("columnas inesperadas en fraude")
        rows = list(reader)
    try:
        labels = np.array([int(row[FRAUD_LABEL]) for row in rows])
    except (TypeError, ValueError) as error:
        raise ValueError("flagged_fraud debe contener enteros") from error
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("flagged_fraud debe contener 0 o 1")
    names = list(FRAUD_FEATURES) + [FRAUD_TARGET]
    try:
        values = np.array([[float(row[name]) if row[name].strip() else np.nan
                            for name in names] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("fraude contiene valores no numéricos") from error
    return np.arange(len(rows)), values, labels


def digit_test_inputs(path):
    """Leer imágenes de test sin interpretar ni devolver sus etiquetas."""
    images = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != {"label", "image"}:
            raise ValueError("columnas inesperadas en test de dígitos")
        for row_number, row in enumerate(reader, start=2):
            try:
                image = np.asarray(ast.literal_eval(row["image"]), dtype=np.float32)
            except (SyntaxError, TypeError, ValueError) as error:
                raise ValueError(f"imagen de test inválida en la fila {row_number}") from error
            if image.shape != (784,) or not np.isfinite(image).all():
                raise ValueError(f"imagen de test inválida en la fila {row_number}")
            images.append(image)
    if not images:
        raise ValueError("digits_test.csv está vacío")
    return np.stack(images)


def duplicates(X, y):
    """Contar entradas idénticas y grupos con objetivos distintos, sin borrarlos."""
    groups = {}
    for x, target in zip(X, np.asarray(y).reshape(-1)):
        key = tuple(x)
        if key not in groups:
            groups[key] = [0, set()]
        groups[key][0] += 1
        groups[key][1].add(float(target))
    repeated = [group for group in groups.values() if group[0] > 1]
    return dict(duplicate_input_groups=len(repeated),
                duplicate_input_rows=sum(group[0] for group in repeated),
                extra_input_copies=sum(group[0] - 1 for group in repeated),
                conflicting_target_groups=sum(len(group[1]) > 1 for group in repeated))


def train_test_overlap(X_train, X_test):
    """Contar entradas idénticas entre splits sin consultar objetivos de test."""
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("training y test deben ser matrices con igual cantidad de entradas")
    dtype = np.result_type(X_train.dtype, X_test.dtype)
    train = np.ascontiguousarray(X_train, dtype=dtype)
    test = np.ascontiguousarray(X_test, dtype=dtype)
    train_keys = [row.tobytes() for row in train]
    test_keys = [row.tobytes() for row in test]
    shared = set(train_keys) & set(test_keys)
    return dict(shared_input_groups=len(shared),
                training_rows_in_shared_groups=sum(key in shared for key in train_keys),
                test_rows_in_shared_groups=sum(key in shared for key in test_keys))


def save_figure(fig, output, name):
    fig.savefig(output / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def analyze_fraud(data_dir, output):
    indices, data, labels = fraud_data(data_dir / "fraud_dataset.csv")
    names = list(FRAUD_FEATURES) + [FRAUD_TARGET]
    stats = [describe(name, data[:, i], unit) for i, (name, unit) in enumerate(zip(names, UNITS))]
    for row in stats:
        row["documented_type"] = "integer" if row["variable"] in INTEGER_FEATURES else "float"
        row["analysis_dtype"] = str(data.dtype)
    write_csv(output / "fraud-statistics.csv", stats)
    write_csv(output / "fraud-training-rows.csv", [{"csv_row": int(i + 2)} for i in indices])
    counts = [{"class": int(label), "count": int((labels == label).sum()),
               "fraction": float((labels == label).mean())} for label in (0, 1)]
    write_csv(output / "fraud-class-balance.csv", counts)

    quality = []
    for i, name in enumerate(names):
        column = data[:, i]
        finite = np.isfinite(column)
        quality.append(dict(variable=name, negative=int((column[finite] < 0).sum()),
                            non_integer=(int((column[finite] % 1 != 0).sum())
                                         if name in INTEGER_FEATURES else ""),
                            nonpositive=(int((column[finite] <= 0).sum())
                                         if name in {"quantity_purchased", "device_screen_resolution"} else ""),
                            above_one=(int((column[finite] > 1).sum()) if name == FRAUD_TARGET else "")))
    write_csv(output / "fraud-semantic-checks.csv", quality)

    # Escalas independientes: las entradas se muestran en sus unidades originales.
    for kind in ("histograms", "boxplots"):
        fig, axes = plt.subplots(5, 2, figsize=(12, 14), constrained_layout=True)
        for i, ax in enumerate(axes.flat):
            column = data[:, i]
            column = column[np.isfinite(column)]
            if len(column):
                if kind == "histograms":
                    ax.hist(column, bins=30, color="#2878a0", edgecolor="white")
                    ax.set_ylabel("Cantidad")
                else:
                    ax.boxplot(column, vert=False, whis=1.5, flierprops={"markersize": 2})
                    ax.set_yticks([])
            ax.set_title(names[i], fontsize=10)
            ax.set_xlabel(UNITS[i])
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.locator_params(axis="x", nbins=4)
        title = "histogramas" if kind == "histograms" else "boxplots"
        fig.suptitle(f"Fraude · dataset completo ({len(indices)} filas) · {title}")
        save_figure(fig, output, f"fraud-{kind}.png")

    # Correlaciones por pares finitos; la etiqueta real nunca participa.
    correlation = np.full((len(names), len(names)), np.nan)
    for i in range(len(names)):
        for j in range(len(names)):
            valid = np.isfinite(data[:, i]) & np.isfinite(data[:, j])
            a, b = data[valid, i], data[valid, j]
            if len(a) > 1 and a.std() > 0 and b.std() > 0:
                correlation[i, j] = np.corrcoef(a, b)[0, 1]
    write_csv(output / "fraud-correlations.csv",
              [dict(variable=name, **dict(zip(names, correlation[i]))) for i, name in enumerate(names)])
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    plot = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), names, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{correlation[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(correlation[i,j]) > 0.65 else "black")
    fig.colorbar(plot, ax=ax, label="Correlación lineal de Pearson")
    ax.set_title("Fraude · entradas y objetivo BigModel · dataset completo")
    save_figure(fig, output, "fraud-correlations.png")

    complete = np.isfinite(data).all(axis=1)
    return dict(rows=len(indices), statistics=stats, classes=counts, semantic_checks=quality,
                constant_features=[row["variable"] for row in stats if row["unique"] == 1],
                duplicate_rows_analyzed=int(complete.sum()),
                duplicates=duplicates(data[complete, :-1], data[complete, -1]))


def analyze_digits(data_dir, output, seed):
    # Las etiquetas y estadísticas provienen sólo de training.
    X, y = _load_digit_file(data_dir / "digits.csv")
    X_test = digit_test_inputs(data_dir / "digits_test.csv")
    stats = [describe(f"pixel_{i}", X[:, i], "intensidad original") for i in range(X.shape[1])]
    write_csv(output / "digits-pixel-statistics.csv", stats)
    global_stats = describe("all_pixels", X.ravel(), "intensidad original")
    write_csv(output / "digits-global-statistics.csv", [global_stats])
    counts = [{"class": label, "count": int((y == label).sum()),
               "fraction": float((y == label).mean())} for label in range(10)]
    write_csv(output / "digits-class-balance.csv", counts)
    means, stds = X.mean(axis=0, dtype=np.float64), X.std(axis=0, dtype=np.float64)
    constant = np.array([row["unique"] == 1 for row in stats])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(range(10), [row["count"] for row in counts], color="#2878a0")
    for row in counts:
        axes[0].annotate(str(row["count"]), (row["class"], row["count"]),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(row["count"] for row in counts) * 1.15)
    axes[0].set(title="Balance de clases", xlabel="Dígito", ylabel="Cantidad", xticks=range(10))
    axes[1].hist(X.ravel(), bins=40, color="#2878a0")
    axes[1].set(title="Distribución de todos los píxeles", xlabel="Intensidad original", ylabel="Cantidad")
    axes[1].set_yscale("log")
    fig.suptitle(f"Dígitos · training ({len(X)} imágenes) · histograma con eje Y logarítmico")
    save_figure(fig, output, "digits-distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for ax, matrix, title in zip(axes, [means, stds, constant],
                                 ["Media por píxel", "Desvío por píxel", "Píxeles constantes"]):
        plot = ax.imshow(matrix.reshape(28, 28), cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Columna"); ax.set_ylabel("Fila")
        fig.colorbar(plot, ax=ax, shrink=0.7)
    save_figure(fig, output, "digits-pixel-maps.png")

    rng = np.random.default_rng(seed)
    intensity_min, intensity_max = float(X.min()), float(X.max())
    fig, axes = plt.subplots(3, 10, figsize=(13, 5), constrained_layout=True)
    selected = []
    for label in range(10):
        available = np.flatnonzero(y == label)
        examples = rng.choice(available, min(3, len(available)), replace=False)
        for row in range(3):
            ax = axes[row, label]; ax.axis("off")
            if row < len(examples):
                index = int(examples[row])
                ax.imshow(X[index].reshape(28, 28), cmap="gray", vmin=intensity_min, vmax=intensity_max)
                ax.set_title(f"{label} · fila {index + 2}", fontsize=8)
                selected.append({"class": label, "csv_row": index + 2})
            elif row == 0:
                ax.text(0.5, 0.5, f"Dígito {label}\nSin muestras", ha="center", va="center",
                        transform=ax.transAxes, color="#9d3b27")
    fig.suptitle(f"Ejemplos de training · selección reproducible, semilla {seed}")
    save_figure(fig, output, "digits-examples.png")
    write_csv(output / "digits-example-rows.csv", selected)
    return dict(rows=len(X), pixels_per_image=X.shape[1], dtype=str(X.dtype),
                statistics=global_stats, classes=counts,
                absent_classes=[row["class"] for row in counts if row["count"] == 0],
                zero_pixel_fraction=float((X == 0).mean()),
                constant_pixels=int(constant.sum()),
                zero_constant_pixels=int((constant & (means == 0)).sum()),
                blank_images=int(np.all(X == 0, axis=1).sum()),
                constant_images=int((np.ptp(X, axis=1) == 0).sum()),
                duplicates=duplicates(X, y),
                train_test_overlap=train_test_overlap(X, X_test))


def run(data_dir, output, seed=0):
    data_dir, output = Path(data_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("usar una carpeta de salida nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    metadata = dict(seed=seed, iqr_factor=1.5,
                    std_ddof=0, percentiles=[0, 1, 25, 50, 75, 99, 100],
                    python=platform.python_version(), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, sha256={})
    for name in ("fraud_dataset.csv", "digits.csv", "digits_test.csv"):
        metadata["sha256"][name] = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
    print("Analizando el dataset completo de fraude...", flush=True)
    fraud = analyze_fraud(data_dir, output)
    print("Analizando digits.csv...", flush=True)
    digits = analyze_digits(data_dir, output, seed)
    result = dict(metadata=metadata, fraud=fraud, digits=digits)
    overlap_rows = [{"dataset": "digits", **digits["train_test_overlap"]}]
    # LF evita que Git interprete el CR propio del dialecto CSV como whitespace.
    write_csv(output / "train-test-overlap.csv", overlap_rows, lineterminator="\n")
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["# EDA: resumen generado", "",
             "Generado por `scripts/analyze_training.py`. No se entrenan modelos ni se transforman datos.", "",
             f"- Fraude: {fraud['rows']} filas del dataset completo; sin split en esta etapa.",
             f"- Dígitos: {digits['rows']} imágenes de {digits['pixels_per_image']} píxeles.",
             "- `digits_test.csv` se abre sólo para comparar imágenes exactas; no se interpretan sus etiquetas.",
             "- No se abre `more_digits.csv`.",
             "- En fraude se analiza el CSV completo, como pide la etapa inicial de aprendizaje del ejercicio 1.",
             "- Estadísticos sobre valores finitos; faltantes/NaN e infinitos se cuentan por separado.",
             "- Desvío descriptivo con ddof=0; cuartiles/percentiles con interpolación lineal de NumPy.",
             "- Correlaciones por pares finitos; duplicados de fraude sólo entre filas numéricas completas.",
             "- El loader de dígitos comprueba columnas, etiquetas 0..9 e imágenes finitas de 784 píxeles;",
             "  interrumpe el análisis ante datos inválidos, sin corregirlos ni omitirlos.",
             "- Histogramas: 30 intervalos para fraude, 40 para píxeles. Son opciones gráficas, no transformaciones.",
             "- Regla IQR: candidatos fuera de Q1−1,5×IQR y Q3+1,5×IQR; no implica datos erróneos.",
             "- Chequeos semánticos de fraude: no negativos; cantidades/resolución positivas; enteros según documentación;",
             "  BigModel en [0,1]. Son alertas para revisar, no filtros aplicados.", "",
             "## Fraude: distribución y escala", "",
             "| Variable | Mínimo | Mediana | Máximo | Desvío | Faltantes/NaN | Infinitos | Candidatos IQR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    for row in fraud["statistics"]:
        keys = ["min", "median", "max", "std", "missing_or_nan", "infinite", "iqr_candidates"]
        lines.append("| " + row["variable"] + " | " + " | ".join(number(row[k]) for k in keys) + " |")
    for name, section in [("Fraude", fraud), ("Dígitos", digits)]:
        lines += ["", f"## {name}: balance y duplicados", "",
                  "| Clase | Cantidad | Porcentaje |", "|---|---:|---:|"]
        lines += [f"| {r['class']} | {r['count']} | {100*r['fraction']:.2f}% |" for r in section["classes"]]
        d = section["duplicates"]
        lines += ["", f"Entradas duplicadas: {d['duplicate_input_groups']} grupos, {d['duplicate_input_rows']} filas involucradas,",
                  f"{d['extra_input_copies']} copias adicionales y {d['conflicting_target_groups']} grupos con objetivos distintos."]
        if "train_test_overlap" in section:
            overlap = section["train_test_overlap"]
            lines += [f"Solapamiento training/test: {overlap['shared_input_groups']} grupos exactos;",
                      f"{overlap['training_rows_in_shared_groups']} filas de training y {overlap['test_rows_in_shared_groups']} de test involucradas."]
    lines += ["", f"Píxeles constantes: {digits['constant_pixels']}; constantes en cero: {digits['zero_constant_pixels']}.",
              f"Clases ausentes en training: {digits['absent_classes']}. Píxeles en cero: {100*digits['zero_pixel_fraction']:.2f}%.",
              f"Imágenes con todos los píxeles iguales: {digits['constant_images']}; completamente en cero: {digits['blank_images']}.",
              f"Rango observado de píxeles: [{digits['statistics']['min']}, {digits['statistics']['max']}].", "",
              "## Gráficos", ""]
    for name in ("fraud-histograms", "fraud-boxplots", "fraud-correlations", "digits-distributions",
                 "digits-pixel-maps", "digits-examples"):
        lines += [f"![{name}]({name}.png)", ""]
    lines += ["Los CSV conservan el detalle completo; `summary.json` registra configuración, versiones y hashes.",
              "Las decisiones de tratamiento requieren revisar estos resultados; no se aplican automáticamente."]
    (output / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe guardado en {output / 'resumen.md'}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.data, args.output, args.seed)

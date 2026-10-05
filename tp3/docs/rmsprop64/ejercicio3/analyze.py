"""Consolidar todas las etapas del ejercicio 3 partiendo de RMSProp-64."""

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize import aggregate, best_row, fixed_row, history, write_csv, RESULTS

HERE = Path(__file__).resolve().parent
SEEDS = range(5)
METRICS = ("accuracy", "macro_f1", "f1_5", "f1_8")

# (etapa, candidato, carpeta, prefijo de corrida, criterio)
CANDIDATES = (
    ("1. Baseline", "Receta del ej. 2 (RMSProp-64, MSE)",
     "02-baseline-five-seeds", "baseline-exercise2-new-data", "best"),
    ("2. Softmax + CE", "η = 0,01", "04-softmax-finalists-five-seeds",
     "softmax-ce-lr-001", "best"),
    ("2. Softmax + CE", "η = 0,03", "04-softmax-finalists-five-seeds",
     "softmax-ce-lr-003", "best"),
    ("3. L2", "λ = 1e-4", "06-l2-five-seeds", "softmax-ce-l2-1e-4", "best"),
    ("4. Traslaciones", "1 px, p = 0,5, con L2", "08-translation-five-seeds",
     "translation-shift-1-p-05", "best"),
    ("4. Traslaciones", "1 px, p = 0,75, con L2", "08-translation-five-seeds",
     "translation-shift-1-p-075", "best"),
    ("4b. Revalidar L2", "1 px, p = 0,5, sin L2",
     "08b-translation-without-l2-five-seeds", "translation-shift-1-p-05", "best"),
    ("4b. Revalidar L2", "1 px, p = 0,75, sin L2",
     "08b-translation-without-l2-five-seeds", "translation-shift-1-p-075", "best"),
    ("5. 300 épocas", "η constante, mejor checkpoint", "09-translation-300-epochs",
     "translation-shift-1-p-05-300", "best"),
    ("5. 300 épocas", "η constante, época 300", "09-translation-300-epochs",
     "translation-shift-1-p-05-300", 300),
    ("6. Schedule", "0,01 → 0,003 → 0,001, época 300",
     "11-learning-rate-schedule-five-seeds", "translation-step-decay", 300),
    ("7. Rotaciones", "±4°, p = 0,5, época 300", "13-small-rotation-five-seeds",
     "rotation-4deg-p-05", 300),
    ("8. Arquitectura", "[784, 96, 10], época 300", "14b-architecture-five-seeds",
     "wide-96", 300),
    ("8. Arquitectura", "[784, 128, 10], época 300", "14b-architecture-five-seeds",
     "wide-128", 300),
)

DECISIONS = (
    ("Softmax + CE (η = 0,01) vs baseline", "02-baseline-five-seeds",
     "baseline-exercise2-new-data", "best", "04-softmax-finalists-five-seeds",
     "softmax-ce-lr-001", "best", "se conserva"),
    ("L2 1e-4 vs sin L2", "04-softmax-finalists-five-seeds", "softmax-ce-lr-001",
     "best", "06-l2-five-seeds", "softmax-ce-l2-1e-4", "best", "se conserva"),
    ("Traslación p = 0,5 vs control con L2", "06-l2-five-seeds",
     "softmax-ce-l2-1e-4", "best", "08-translation-five-seeds",
     "translation-shift-1-p-05", "best", "se conserva"),
    ("Con L2 vs sin L2, ambos con traslación p = 0,5",
     "08b-translation-without-l2-five-seeds", "translation-shift-1-p-05", "best",
     "08-translation-five-seeds", "translation-shift-1-p-05", "best",
     "se retira L2"),
    ("Schedule vs η constante (época 300)", "09-translation-300-epochs",
     "translation-shift-1-p-05-300", 300, "11-learning-rate-schedule-five-seeds",
     "translation-step-decay", 300, "se conserva"),
    ("Rotación ±4° vs sin rotación (época 300)",
     "11-learning-rate-schedule-five-seeds", "translation-step-decay", 300,
     "13-small-rotation-five-seeds", "rotation-4deg-p-05", 300, "se descarta"),
    ("96 vs 64 neuronas (época 300)", "11-learning-rate-schedule-five-seeds",
     "translation-step-decay", 300, "14b-architecture-five-seeds", "wide-96", 300,
     "se conserva 96"),
    ("128 vs 96 neuronas (época 300)", "14b-architecture-five-seeds", "wide-96", 300,
     "14b-architecture-five-seeds", "wide-128", 300,
     "empate: se prefiere 96"),
)

# Medias de validation del ejercicio 3 original (Adam-128), para comparar.
ORIGINAL = {
    "1. Baseline": 0.9547, "2. Softmax + CE": 0.9642, "4. Traslaciones": 0.9754,
    "6. Schedule": 0.9782, "7. Rotaciones": 0.9793, "5-fold OOF": 0.9787,
    "Test": 0.9752,
}


def rows_for(stage, prefix, criterion):
    names = [f"{prefix}-seed-{seed}" for seed in SEEDS]
    if criterion == "best":
        return [best_row(stage, name) for name in names]
    return [fixed_row(stage, name, criterion) for name in names]


def candidate_table():
    table = []
    for etapa, candidate, stage, prefix, criterion in CANDIDATES:
        rows = rows_for(stage, prefix, criterion)
        summary = aggregate(rows, METRICS)
        table.append({
            "etapa": etapa, "candidato": candidate,
            "criterio": "mejor checkpoint" if criterion == "best" else f"época {criterion}",
            **{key: round(value, 4) for key, value in summary.items()},
        })
    return table


def decision_table():
    table = []
    for label, s0, p0, c0, s1, p1, c1, decision in DECISIONS:
        before, after = rows_for(s0, p0, c0), rows_for(s1, p1, c1)
        row = {"comparacion": label, "decision": decision}
        for metric in ("accuracy", "macro_f1"):
            differences = [b[metric] - a[metric] for a, b in zip(before, after)]
            mean = statistics.mean(differences)
            margin = 2.776 * statistics.stdev(differences) / np.sqrt(len(differences))
            row.update({
                f"{metric}_diff_mean": round(mean, 4),
                f"{metric}_ic95_low": round(mean - margin, 4),
                f"{metric}_ic95_high": round(mean + margin, 4),
                f"{metric}_semillas_mejoran": sum(value > 0 for value in differences),
            })
        table.append(row)
    return table


def stability_table():
    table = []
    for label, stage, prefix in (
            ("η constante", "09-translation-300-epochs", "translation-shift-1-p-05-300"),
            ("Schedule", "11-learning-rate-schedule-five-seeds", "translation-step-decay")):
        rows = rows_for(stage, prefix, 300)
        table.append({"configuracion": label, **{
            key: round(value, 5) for key, value in aggregate(rows, (
                "accuracy_sd_last_50", "macro_f1_sd_last_50", "loss_sd_last_50",
                "accuracy_mean_last_50")).items() if key.endswith("_mean")}})
    return table


def plot_curves(output):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = (
        ("L2 con y sin traslaciones", (
            ("Con L2", "08-translation-five-seeds", "translation-shift-1-p-05", "#E67E22"),
            ("Sin L2", "08b-translation-without-l2-five-seeds",
             "translation-shift-1-p-05", "#1F5D8F"))),
        ("η constante vs schedule", (
            ("η constante", "09-translation-300-epochs",
             "translation-shift-1-p-05-300", "#E67E22"),
            ("Schedule", "11-learning-rate-schedule-five-seeds",
             "translation-step-decay", "#1F5D8F"))),
        ("Ancho de la capa oculta", (
            ("64", "11-learning-rate-schedule-five-seeds", "translation-step-decay",
             "#9FB3C8"),
            ("96", "14b-architecture-five-seeds", "wide-96", "#1F5D8F"),
            ("128", "14b-architecture-five-seeds", "wide-128", "#E67E22"))),
    )
    for axis, (title, series) in zip(axes, panels):
        for label, stage, prefix, color in series:
            matrix = np.array([[float(row["validation_accuracy"])
                                for row in history(RESULTS / stage / f"{prefix}-seed-{seed}")]
                               for seed in SEEDS])
            epochs = np.arange(matrix.shape[1])
            mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
            axis.plot(epochs, mean, color=color, lw=1.4, label=label)
            axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Época")
        axis.set_ylim(0.93, 0.985)
        axis.set_xlim(20, None)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Accuracy de validation (media ± sd, 5 semillas)")
    figure.tight_layout()
    figure.savefig(output / "decision-curves.png", dpi=170)
    plt.close(figure)


def plot_evolution(output, kfold_accuracy, test_accuracy):
    steps = [
        ("Baseline", 0.9590, ORIGINAL["1. Baseline"]),
        ("Softmax + CE", 0.9620, ORIGINAL["2. Softmax + CE"]),
        ("Traslaciones", 0.9736, ORIGINAL["4. Traslaciones"]),
        ("Schedule\n(época 300)", 0.9733, ORIGINAL["6. Schedule"]),
        ("Rotaciones (orig.) /\n96 neuronas (variante)", 0.9773, ORIGINAL["7. Rotaciones"]),
        ("5-fold OOF", kfold_accuracy, ORIGINAL["5-fold OOF"]),
        ("Test", test_accuracy, ORIGINAL["Test"]),
    ]
    x = np.arange(len(steps))
    figure, axis = plt.subplots(figsize=(10, 4.3))
    axis.plot(x, [s[2] for s in steps], "o-", color="#9FB3C8", label="Original (Adam-128)")
    axis.plot(x, [s[1] for s in steps], "o-", color="#1F5D8F", label="Variante RMSProp-64")
    axis.axhline(0.98, color="#C0392B", ls="--", lw=1, label="Objetivo 98 %")
    for xi, (_, value, _) in zip(x, steps):
        decimals = 3 if xi == len(steps) - 1 else 2
        axis.annotate(f"{value * 100:.{decimals}f}".replace(".", ","), (xi, value),
                      textcoords="offset points", xytext=(0, 8), ha="center",
                      fontsize=8, color="#102A43")
    axis.set_xticks(x)
    axis.set_xticklabels([s[0] for s in steps], fontsize=9)
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.95, 0.985)
    axis.set_xlim(-0.4, len(steps) - 0.6)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Evolución de la accuracy (validation media de 5 semillas, luego 5-fold y test)")
    figure.tight_layout()
    figure.savefig(output / "evolution.png", dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_table()
    write_csv(args.output / "candidates.csv", candidates)
    write_csv(args.output / "paired-decisions.csv", decision_table())
    write_csv(args.output / "schedule-stability.csv", stability_table())
    search_rows = []
    for stage, criterion in (("03-softmax-learning-rate", "best"),
                             ("03b-softmax-learning-rate-upper", "best"),
                             ("05-l2-search", "best"),
                             ("07-translation-augmentation-search", "best"),
                             ("07b-translation-search-without-l2", "best"),
                             ("12-small-rotation-search", 300),
                             ("14-architecture-search", 300),
                             ("15-batch-size-search", 300),
                             ("15b-batch-size-search-96", 300)):
        for name in sorted(path.name for path in (RESULTS / stage).iterdir()
                           if (path / "history.csv").exists()):
            row = (best_row(stage, name) if criterion == "best"
                   else fixed_row(stage, name, criterion))
            search_rows.append({"etapa": stage, **{
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in row.items() if "last_50" not in key}})
    write_csv(args.output / "single-seed-searches.csv", search_rows)
    plot_curves(args.output)
    import json
    kfold = json.loads((args.output / "16-kfold" / "kfold-summary.json").read_text())
    final = json.loads((RESULTS / "17-final-evaluation" / "summary.json").read_text())
    plot_evolution(args.output, kfold["out_of_fold_accuracy"], final["test_accuracy"])
    for row in candidates:
        print(row["etapa"], row["candidato"], row["accuracy_mean"], row["macro_f1_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

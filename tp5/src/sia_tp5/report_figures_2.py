"""Figuras y tablas del ejercicio 2 a partir de results/vae-code-study/.

Uso: uv run python -m sia_tp5.report_figures_2
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .code_dataset import (decode_patterns, function_parameters,  # noqa: E402
                           load_code_dataset, semantic_signature)
from .variational_autoencoder import VariationalAutoencoder  # noqa: E402
from .vae_code_study import classify, reconstruction  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "vae-code-study"
FIGS = ROOT / "docs" / "ejs-figs"
SUMMARY = FIGS / "2-vae-estudio.csv"
REFERENCE = json.loads((FIGS / "2-vae-estudio.referencia.json").read_text())
CHOSEN = "beta003-l32"
CONTROL = "beta0-l16-s0"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


def _summaries() -> dict:
    with SUMMARY.open(encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def _style(axis) -> None:
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_beta(rows: dict, path: Path) -> None:
    betas = [("0", "beta0-l16-s0"), ("0,01", "beta001-l16-s0"), ("0,03", "beta003-l16-s0"),
             ("0,1", "beta01-l16-s0"), ("0,3", "beta03-l16-s0"), ("1", "beta10-l16-s0")]
    labels = [b for b, _ in betas]
    x = np.arange(len(betas))
    get = lambda key: np.array([float(rows[name][key]) for _, name in betas])  # noqa: E731
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(x, get("validation_exact") * 100, marker="o", color=BLUE, linewidth=2,
              label="reconstrucción exacta (validación)")
    left.plot(x, get("prior_grammar") * 100, marker="o", color=ORANGE, linewidth=2,
              label="muestras del prior válidas")
    left.set_ylim(0, 105)
    left.set_ylabel("%", color=TEXT_SECONDARY)
    left.set_title("Reconstruir contra generar", fontsize=10, color=TEXT_PRIMARY)
    left.legend(frameon=False, fontsize=9, loc="lower center")
    for key, color, label in (("tv_operator_pairs", BLUE, "pares de operadores"),
                              ("tv_constants", ORANGE, "constantes")):
        right.plot(x, get(key), marker="o", color=color, linewidth=2, label=label)
        right.axhline(REFERENCE[key], color=color, linestyle=":", linewidth=1.2)
    right.text(x[-1] + 0.1, REFERENCE["tv_constants"], "ideal", color=ORANGE,
               fontsize=8, va="bottom", ha="right")
    right.text(x[-1] + 0.1, REFERENCE["tv_operator_pairs"], "ideal", color=BLUE,
               fontsize=8, va="bottom", ha="right")
    right.set_ylim(0, None)
    right.set_ylabel("distancia a la distribución de los datos", color=TEXT_SECONDARY)
    right.set_title("¿Lo generado se distribuye como los datos? (menor es mejor)",
                    fontsize=10, color=TEXT_PRIMARY)
    right.legend(frameon=False, fontsize=9, loc="upper right")
    for axis in (left, right):
        axis.set_xticks(x, labels)
        axis.set_xlabel("beta (peso de la KL)", color=TEXT_SECONDARY)
        _style(axis)
    fig.suptitle("Paso 1: peso de la KL (latente 16, semilla 0)", color=TEXT_PRIMARY,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_latent(rows: dict, path: Path) -> None:
    dims = [2, 8, 16, 32]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [("validation_exact", "Reconstrucción exacta\n(validación, %)", 100),
               ("prior_grammar", "Muestras del prior válidas (%)", 100),
               ("tv_constants", "Distancia a la distribución de constantes\n"
                                "(menor es mejor)", 1)]
    for axis, (key, title, scale) in zip(axes, metrics):
        for beta, color in (("003", ORANGE), ("01", BLUE)):
            values = [float(rows[f"beta{beta}-l{d}-s0"][key]) * scale for d in dims]
            axis.plot(range(len(dims)), values, marker="o", color=color, linewidth=2,
                      label=f"beta {'0,03' if beta == '003' else '0,1'}")
        if key == "tv_constants":
            axis.axhline(REFERENCE["tv_constants"], color=TEXT_SECONDARY, linestyle=":",
                         label="generador ideal")
        axis.set_xticks(range(len(dims)), [str(d) for d in dims])
        axis.set_xlabel("dimensión del latente", color=TEXT_SECONDARY)
        axis.set_title(title, fontsize=10, color=TEXT_PRIMARY)
        _style(axis)
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Paso 2: dimensión del espacio latente (semilla 0)", color=TEXT_PRIMARY,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_finalists(rows: dict, path: Path) -> None:
    finalists = [("beta01-l16", "beta 0,1\nlatente 16"), ("beta03-l16", "beta 0,3\nlatente 16"),
                 ("beta01-l32", "beta 0,1\nlatente 32"), ("beta003-l32", "beta 0,03\nlatente 32")]
    metrics = [("tv_operator_pairs", "Distancia a la distribución\nde pares de operadores"),
               ("tv_constants", "Distancia a la distribución\nde constantes"),
               ("prior_novel_semantic", "Muestras que calculan\nalgo nuevo (fracción)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for axis, (key, title) in zip(axes, metrics):
        for index, (prefix, _) in enumerate(finalists):
            values = np.array([float(rows[f"{prefix}-s{s}"][key]) for s in range(3)])
            color = BLUE if prefix == CHOSEN else TEXT_SECONDARY
            axis.bar(index, values.mean(), yerr=values.std(ddof=1), color=color,
                     alpha=0.9 if prefix == CHOSEN else 0.45, width=0.6, capsize=3)
            axis.scatter(np.full(3, index), values, s=12, color=TEXT_PRIMARY, zorder=3)
        axis.axhline(REFERENCE[key], color=ORANGE, linestyle="--", linewidth=1.5,
                     label="generador ideal (uniforme)")
        axis.set_xticks(range(len(finalists)), [label for _, label in finalists],
                        fontsize=8.5)
        axis.set_title(title, fontsize=10, color=TEXT_PRIMARY)
        _style(axis)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Paso 3: finalistas con 3 semillas (barra = media, puntos = semillas)",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_losses(path: Path) -> None:
    with (RESULTS / f"{CHOSEN}-s0" / "history.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    epochs = np.array([int(r["epoch"]) for r in history])
    reconstruction_loss = np.array([float(r["reconstruction_loss"]) for r in history])
    kl = np.array([float(r["kl_loss"]) for r in history])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    left.plot(epochs, reconstruction_loss, color=BLUE, label="reconstrucción")
    left.plot(epochs, 0.03 * kl, color=ORANGE, label="beta · KL")
    left.set_yscale("log")
    left.set_title("Términos de la pérdida", fontsize=10, color=TEXT_PRIMARY)
    left.legend(frameon=False)
    right.plot(epochs, kl, color=AQUA)
    right.set_title("KL sin ponderar", fontsize=10, color=TEXT_PRIMARY)
    for axis in (left, right):
        axis.set_xlabel("época", color=TEXT_SECONDARY)
        _style(axis)
    fig.suptitle("Entrenamiento del VAE elegido (beta 0,03, latente 32, semilla 0)",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _pca(reference: np.ndarray):
    """Proyección a 2 componentes principales, solo para visualizar."""
    center = reference.mean(axis=0)
    _, _, components = np.linalg.svd(reference - center, full_matrices=False)
    return lambda values: (values - center) @ components[:2].T


def plot_latent_space(dataset, path: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    stats = {}
    for axis, name, title in ((axes[0], CONTROL, "Sin KL (beta 0, latente 16)"),
                              (axes[1], f"{CHOSEN}-s0", "VAE elegido (beta 0,03, latente 32)")):
        model = VariationalAutoencoder.load(RESULTS / name / "model.npz")
        mu, _ = model.encode_distribution(dataset.train_patterns)
        z = np.random.default_rng(44).standard_normal((1000, mu.shape[1]))
        generated = decode_patterns(model.decode(z))
        known = set(dataset.sources)
        signatures = {semantic_signature(s) for s in dataset.sources}
        valid = np.array([classify(s, known, signatures)["grammar"] for s in generated])
        project = _pca(mu)
        data_2d, prior_2d = project(mu), project(z)
        axis.scatter(data_2d[:, 0], data_2d[:, 1], s=5, color=TEXT_SECONDARY, alpha=0.35,
                     linewidth=0, label="funciones de entrenamiento (mu)")
        axis.scatter(prior_2d[valid, 0], prior_2d[valid, 1], s=9, color=BLUE, alpha=0.7,
                     linewidth=0, label="z ~ N(0, I): genera función válida")
        axis.scatter(prior_2d[~valid, 0], prior_2d[~valid, 1], s=9, color=RED, alpha=0.8,
                     linewidth=0, label="z ~ N(0, I): genera algo inválido")
        axis.set_title(f"{title}\nmuestras válidas: {valid.mean() * 100:.1f} %".replace(".", ","),
                       fontsize=10, color=TEXT_PRIMARY)
        axis.set_xlabel("componente principal 1", color=TEXT_SECONDARY)
        axis.set_ylabel("componente principal 2", color=TEXT_SECONDARY)
        _style(axis)
        stats[name] = {"mu_std": float(mu.std()), "valid": float(valid.mean())}
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right", markerscale=2)
    fig.suptitle("Dónde están los datos y dónde cae el prior N(0, I)\n"
                 "proyección PCA de los códigos, solo para visualizar",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return stats


def final_tables(dataset) -> dict:
    """Test con las 10 semillas, muestras e interpolación del modelo semilla 0."""
    rows = _summaries()
    seeds = []
    for seed in range(10):
        name = f"{CHOSEN}-s{seed}"
        model = VariationalAutoencoder.load(RESULTS / name / "model.npz")
        test = reconstruction(model, dataset.test_patterns, dataset.test_sources)
        seeds.append({"seed": seed, **{k: float(rows[name][k]) for k in (
            "validation_exact", "prior_grammar", "prior_novel_text", "prior_novel_semantic",
            "tv_operator_pairs", "tv_constants", "kl")},
            "prior_unique_valid": int(rows[name]["prior_unique_valid"]),
            "test_exact": test["exact"], "test_character_accuracy": test["character_accuracy"]})
    model = VariationalAutoencoder.load(RESULTS / f"{CHOSEN}-s0" / "model.npz")
    with (RESULTS / f"{CHOSEN}-s0" / "prior-samples.csv").open(encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))[:12]
    train = list(dataset.train_sources)
    first = lambda ops: next(s for s in train  # noqa: E731
                             if function_parameters(s)[0::2] == ops)
    left, right = first(("+", "+")), first(("*", "*"))
    index = {s: i for i, s in enumerate(train)}
    mu, _ = model.encode_distribution(dataset.train_patterns[[index[left], index[right]]])
    known = set(dataset.sources)
    signatures = {semantic_signature(s) for s in dataset.sources}
    interpolation = []
    for t in np.linspace(0, 1, 11):
        source = decode_patterns(model.decode((1 - t) * mu[:1] + t * mu[1:]))[0]
        info = classify(source, known, signatures)
        interpolation.append({"t": round(float(t), 1), "source": source, **info,
                              "in_train": source in set(train)})
    # Segundo recorrido: entre las dos primeras muestras del prior que calculan
    # algo nuevo (regla fijada antes de mirar el resultado).
    z_all = np.random.default_rng(44).standard_normal((1000, mu.shape[1]))
    novel_index = [int(row["sample"]) for row in
                   csv.DictReader((RESULTS / f"{CHOSEN}-s0" / "prior-samples.csv").open())
                   if row["novel_semantic"] == "True"][:2]
    z_a, z_b = z_all[novel_index[0]], z_all[novel_index[1]]
    prior_path = []
    for t in np.linspace(0, 1, 11):
        source = decode_patterns(model.decode(((1 - t) * z_a + t * z_b)[None, :]))[0]
        info = classify(source, known, signatures)
        prior_path.append({"t": round(float(t), 1), "source": source, **info,
                           "in_train": source in set(train)})
    return {"seeds": seeds, "samples": samples, "interpolation": interpolation,
            "prior_interpolation": prior_path, "prior_interpolation_ends": novel_index}


def main() -> None:
    dataset = load_code_dataset(split_seed=17)
    rows = _summaries()
    plot_beta(rows, FIGS / "2-paso1-beta.png")
    plot_latent(rows, FIGS / "2-paso2-latente.png")
    plot_finalists(rows, FIGS / "2-paso3-finalistas.png")
    plot_losses(FIGS / "2-perdidas.png")
    stats = plot_latent_space(dataset, FIGS / "2-latente-prior.png")
    tables = final_tables(dataset)
    tables["latent_stats"] = stats
    (FIGS / "2-final.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(tables, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()

"""Generar gráficos desde una corrida guardada, sin volver a entrenar."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3 import MultilayerPerceptron


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    cases = list(dict.fromkeys(row['case'] for row in summary))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.5 * len(cases)), layout='constrained')
    for ax, case in zip(np.atleast_1d(axes), cases):
        for row in [r for r in summary if r['case'] == case]:
            folder = args.run / f"{case}-seed-{row['seed']}"
            with (folder / 'history.csv').open() as file:
                history = list(csv.DictReader(file))
            ax.plot([int(r['epoch']) for r in history], [float(r['mse']) for r in history],
                    label=f"Semilla {row['seed']} · {'PASS' if row['passed'] else 'FAIL'}")
        ax.set(title=case, xlabel='Épocas', ylabel='MSE')
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle('Aprendizaje real del motor · MSE evaluado al terminar cada época')
    fig.savefig(args.output / 'learning.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, case in zip(axes, ['linear', 'tanh']):
        rows = [r for r in summary if r['case'] == case]
        if not rows:
            continue
        folder = args.run / f"{case}-seed-{rows[0]['seed']}"
        with np.load(folder / 'data.npz') as data:
            X, y = data['X'], data['y']
        ax.scatter(X[:, 0], y[:, 0], color='black', s=18, label='Objetivo')
        for row in rows:
            model = MultilayerPerceptron.load(args.run / f"{case}-seed-{row['seed']}" / 'model.npz')
            ax.plot(X[:, 0], model.predict(X)[:, 0], label=f"Aprendido · semilla {row['seed']}")
        ax.set(title=case, xlabel='Entrada x', ylabel='Salida y')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle('Funciones ajustadas con pesos aprendidos')
    fig.savefig(args.output / 'fits.png', dpi=160)
    plt.close(fig)

    selected = [c for c in ['and', 'xor-2-2-1', 'xor-2-3-2-1'] if c in cases]
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.5), layout='constrained')
    xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 180), np.linspace(-1.6, 1.6, 180))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for ax, case in zip(np.atleast_1d(axes), selected):
        row = next(r for r in summary if r['case'] == case)
        folder = args.run / f"{case}-seed-{row['seed']}"
        model = MultilayerPerceptron.load(folder / 'model.npz')
        predicted = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, np.where(predicted >= 0, 1, -1), levels=[-2, 0, 2],
                    colors=['#e49448', '#318aca'], alpha=0.18)
        if predicted.min() < 0 < predicted.max():
            ax.contour(xx, yy, predicted, levels=[0], colors=['#333333'], linewidths=1.4)
        with np.load(folder / 'data.npz') as data:
            X, y = data['X'], data['y'][:, 0]
        for label, marker, color in [(-1, 's', '#b8641c'), (1, 'o', '#126ca8')]:
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], marker=marker, color=color, s=65,
                       label=f'Objetivo {label:+d}')
        ax.set(title=f"{case} · semilla {row['seed']}", xlabel='Entrada x₁', ylabel='Entrada x₂', aspect='equal')
        ax.legend(fontsize=8)
    fig.suptitle('Fronteras aprendidas · El sombreado es la predicción, las formas son el objetivo')
    fig.savefig(args.output / 'boundaries.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

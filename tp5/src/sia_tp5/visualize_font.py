"""Generar una grilla para verificar visualmente los datos de ``font.h``."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .data import load_font


def plot_font(output: Path) -> None:
    dataset = load_font()
    figure, axes = plt.subplots(4, 8, figsize=(12, 7))
    for axis, image, label, codepoint in zip(
            axes.flat, dataset.images, dataset.labels, dataset.codepoints):
        axis.imshow(image, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{label} (0x{codepoint:02x})", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("32 patrones originales de font.h (7 x 5)", fontsize=14)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/font-original.png"),
        help="ruta de la figura PNG",
    )
    arguments = parser.parse_args()
    plot_font(arguments.output)


if __name__ == "__main__":
    main()

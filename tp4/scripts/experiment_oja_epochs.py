#!/usr/bin/env python3
"""Entrada ejecutable para comparar cantidades de épocas de Oja."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sia_tp4.oja_epoch_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()

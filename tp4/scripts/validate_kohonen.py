#!/usr/bin/env python3
"""Entrada ejecutable para validar Kohonen con datos sintéticos."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sia_tp4.kohonen_synthetic import main  # noqa: E402


if __name__ == "__main__":
    main()

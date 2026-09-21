"""Perceptrones y entrenamiento supervisado del TP3."""

from .models import MultilayerPerceptron, Perceptron
from .training import fit

__all__ = ["MultilayerPerceptron", "Perceptron", "fit"]

"""Immutable genetic representation for the triangle-image domain."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Optional, Tuple


Point = Tuple[float, float]
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]


def _twice_signed_area(vertices: Tuple[Point, Point, Point]) -> float:
    (x1, y1), (x2, y2), (x3, y3) = vertices
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)


@dataclass(frozen=True)
class TriangleGene:
    """One gene: three normalized vertices and one RGBA color."""

    vertices: Tuple[Point, Point, Point]
    color: RGBA

    def __post_init__(self) -> None:
        if len(self.vertices) != 3:
            raise ValueError("a triangle must contain exactly three vertices")
        for vertex in self.vertices:
            if len(vertex) != 2:
                raise ValueError("each vertex must contain exactly two coordinates")
            for coordinate in vertex:
                if not isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
                    raise ValueError("triangle coordinates must be finite and in [0, 1]")

        if abs(_twice_signed_area(self.vertices)) <= 1e-12:
            raise ValueError("triangle vertices must define a positive area")

        if len(self.color) != 4:
            raise ValueError("triangle color must contain RGBA components")
        for component in self.color:
            if isinstance(component, bool) or not isinstance(component, int):
                raise ValueError("RGBA components must be integers")
            if not 0 <= component <= 255:
                raise ValueError("RGBA components must be in [0, 255]")
        if self.color[3] == 0:
            raise ValueError("triangle alpha must be greater than zero")

    def to_dict(self) -> dict:
        return {
            "vertices": [[x, y] for x, y in self.vertices],
            "color": list(self.color),
        }


@dataclass(frozen=True)
class Individual:
    """A candidate solution represented by one fixed-length chromosome."""

    chromosome: Tuple[TriangleGene, ...]
    error: Optional[float] = None
    fitness: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.chromosome:
            raise ValueError("an individual chromosome cannot be empty")
        if (self.error is None) != (self.fitness is None):
            raise ValueError("error and fitness must be both present or both absent")
        if self.error is not None:
            if not isfinite(self.error) or not 0.0 <= self.error <= 1.0:
                raise ValueError("individual error must be finite and in [0, 1]")
            if not isfinite(self.fitness) or not 0.0 < self.fitness <= 1.0:
                raise ValueError("individual fitness must be finite and in (0, 1]")

    def evaluated(self, *, error: float, fitness: float) -> "Individual":
        return replace(self, error=error, fitness=fitness)

    def to_dict(self) -> dict:
        return {
            "triangle_count": len(self.chromosome),
            "triangles": [gene.to_dict() for gene in self.chromosome],
            "error": self.error,
            "fitness": self.fitness,
        }


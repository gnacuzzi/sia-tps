import pytest

from sia_tp2.domain.model import Individual, TriangleGene


def test_triangle_requires_positive_area() -> None:
    with pytest.raises(ValueError, match="positive area"):
        TriangleGene(
            vertices=((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
            color=(10, 20, 30, 255),
        )


def test_triangle_rejects_zero_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        TriangleGene(
            vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            color=(10, 20, 30, 0),
        )


def test_individual_evaluation_returns_new_immutable_value() -> None:
    triangle = TriangleGene(
        vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        color=(10, 20, 30, 255),
    )
    individual = Individual(chromosome=(triangle,))

    evaluated = individual.evaluated(error=0.25, fitness=0.75)

    assert individual.error is None
    assert individual.fitness is None
    assert evaluated.error == 0.25
    assert evaluated.fitness == 0.75


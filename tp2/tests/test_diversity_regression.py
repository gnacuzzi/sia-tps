from sia_tp2.domain.diversity import triangle_population_diversity
from sia_tp2.domain.model import Individual, TriangleGene


def test_large_identical_population_has_exactly_zero_diversity() -> None:
    individual = Individual(
        chromosome=tuple(
            TriangleGene(
                vertices=((0.123, 0.234), (0.567, 0.345), (0.456, 0.789)),
                color=(17, 93, 201, 127),
            )
            for _ in range(100)
        )
    )

    assert triangle_population_diversity((individual,) * 50) == 0.0

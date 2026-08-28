from sia_tp2.domain.initialization import create_random_population


def test_same_seed_reproduces_the_same_population() -> None:
    arguments = {
        "population_size": 4,
        "triangle_count": 3,
        "alpha_range": (10, 200),
        "seed": 42,
    }

    first = create_random_population(**arguments)
    second = create_random_population(**arguments)

    assert first == second
    assert all(len(individual.chromosome) == 3 for individual in first)
    assert all(
        10 <= triangle.color[3] <= 200
        for individual in first
        for triangle in individual.chromosome
    )


def test_different_seeds_change_the_population() -> None:
    first = create_random_population(
        population_size=2, triangle_count=2, alpha_range=(1, 255), seed=1
    )
    second = create_random_population(
        population_size=2, triangle_count=2, alpha_range=(1, 255), seed=2
    )

    assert first != second


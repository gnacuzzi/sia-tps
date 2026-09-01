import random
from dataclasses import dataclass

from sia_tp2.domain.model import Individual, TriangleGene
from sia_tp2.domain.operators import mutate_individual, one_point_crossover
from sia_tp2.ga.selection import select_population
from sia_tp2.ga.survival import select_survivors


@dataclass(frozen=True)
class Candidate:
    name: str
    fitness: float


def _individual(*colors: int) -> Individual:
    return Individual(
        chromosome=tuple(
            TriangleGene(
                vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                color=(color, color, color, 100),
            )
            for color in colors
        )
    )


def test_all_selection_methods_return_requested_members_from_population() -> None:
    population = (
        Candidate("a", 0.2),
        Candidate("b", 0.5),
        Candidate("c", 0.8),
    )
    configurations = {
        "elite": {},
        "roulette": {},
        "universal": {},
        "ranking": {},
        "boltzmann": {
            "initial_temperature": 1.0,
            "final_temperature": 0.1,
            "decay_rate": 0.2,
        },
        "tournament_deterministic": {"tournament_size": 2},
        "tournament_probabilistic": {"threshold": 0.75},
    }

    for method, params in configurations.items():
        selected = select_population(
            population,
            count=5,
            method=method,
            params=params,
            generation=3,
            fitness=lambda item: item.fitness,
            rng=random.Random(4),
        )
        assert len(selected) == 5
        assert all(item in population for item in selected)


def test_one_point_crossover_exchanges_complete_suffixes() -> None:
    first = _individual(10, 20, 30, 40)
    second = _individual(50, 60, 70, 80)

    first_child, second_child = one_point_crossover(
        first, second, probability=1.0, rng=random.Random(2)
    )

    assert len(first_child.chromosome) == 4
    assert len(second_child.chromosome) == 4
    transitions = [
        first_child.chromosome[index] != first.chromosome[index]
        for index in range(4)
    ]
    assert transitions in (
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
    )
    for index in range(4):
        assert {
            first_child.chromosome[index],
            second_child.chromosome[index],
        } == {first.chromosome[index], second.chromosome[index]}


def test_single_gene_global_mutation_changes_exactly_one_triangle() -> None:
    parent = _individual(10, 20, 30)

    child = mutate_individual(
        parent,
        method="single_gene",
        probability=1.0,
        allele_mode="global_resample",
        position_delta=None,
        color_delta=None,
        alpha_delta=None,
        alpha_range=(1, 255),
        rng=random.Random(3),
    )

    changed = sum(
        first != second
        for first, second in zip(parent.chromosome, child.chromosome)
    )
    assert changed == 1
    assert parent == _individual(10, 20, 30)


def test_exclusive_survival_respects_both_offspring_count_cases() -> None:
    parents = tuple(Candidate(f"p{index}", 0.9 - index / 10) for index in range(4))
    few_children = (Candidate("c0", 0.1), Candidate("c1", 0.2))
    many_children = tuple(
        Candidate(f"c{index}", 0.1 + index / 10) for index in range(6)
    )

    def elite(pool, count, generation, rng):
        del generation, rng
        return sorted(pool, key=lambda item: item.fitness, reverse=True)[:count]

    with_few = select_survivors(
        parents,
        few_children,
        population_size=4,
        strategy="exclusive",
        generation=1,
        selector=elite,
        rng=random.Random(0),
    )
    with_many = select_survivors(
        parents,
        many_children,
        population_size=4,
        strategy="exclusive",
        generation=1,
        selector=elite,
        rng=random.Random(0),
    )

    assert with_few[:2] == few_children
    assert all(item in parents for item in with_few[2:])
    assert len(with_many) == 4
    assert all(item in many_children for item in with_many)

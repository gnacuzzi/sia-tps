import random

from sia_tp2.domain.diversity import triangle_population_diversity
from sia_tp2.domain.model import Individual, TriangleGene
from sia_tp2.domain.operators import (
    mutate_individual,
    uniform_crossover,
)
from sia_tp2.ga.selection import deterministic_tournament, elite_selection
from sia_tp2.ga.survival import select_survivors


def _gene(offset: int) -> TriangleGene:
    return TriangleGene(
        vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        color=(offset, offset + 1, offset + 2, 100 + offset),
    )


def _individual(*offsets: int, fitness: float = 0.5) -> Individual:
    return Individual(chromosome=tuple(_gene(value) for value in offsets)).evaluated(
        error=1.0 - fitness,
        fitness=fitness,
    )


def test_deterministic_tournament_selects_best_when_tournament_is_whole_pool() -> None:
    population = (
        _individual(1, fitness=0.2),
        _individual(2, fitness=0.8),
        _individual(3, fitness=0.5),
    )

    selected = deterministic_tournament(
        population,
        count=4,
        tournament_size=3,
        fitness=lambda item: item.fitness,
        rng=random.Random(0),
    )

    assert selected == (population[1],) * 4


def test_uniform_crossover_preserves_loci_and_parental_genes() -> None:
    first = _individual(1, 10, 20)
    second = _individual(30, 40, 50)

    children = uniform_crossover(
        first,
        second,
        probability=1.0,
        swap_probability=0.5,
        rng=random.Random(4),
    )

    for locus in range(3):
        assert {
            children[0].chromosome[locus],
            children[1].chromosome[locus],
        } == {first.chromosome[locus], second.chromosome[locus]}
    assert all(child.error is None and child.fitness is None for child in children)


def test_multigene_probability_one_changes_every_gene_without_mutating_parent() -> None:
    parent = _individual(10, 20, 30)

    child = mutate_individual(
        parent,
        method="multigene_uniform",
        probability=1.0,
        allele_mode="local_delta",
        position_delta=0.1,
        color_delta=10,
        alpha_delta=10,
        alpha_range=(1, 255),
        rng=random.Random(2),
    )

    assert all(
        original != changed
        for original, changed in zip(parent.chromosome, child.chromosome)
    )
    assert parent == _individual(10, 20, 30)
    assert child.error is None and child.fitness is None


def test_additive_survival_with_elite_keeps_best_from_both_sources() -> None:
    parents = (_individual(1, fitness=0.9), _individual(2, fitness=0.2))
    children = (_individual(3, fitness=0.8), _individual(4, fitness=0.1))

    def selector(pool, count, generation, rng):
        del generation, rng
        return elite_selection(pool, count=count, fitness=lambda item: item.fitness)

    survivors = select_survivors(
        parents,
        children,
        population_size=2,
        strategy="additive",
        generation=1,
        selector=selector,
        rng=random.Random(0),
    )

    assert [item.fitness for item in survivors] == [0.9, 0.8]


def test_triangle_diversity_is_zero_only_for_identical_population() -> None:
    first = _individual(1, 2)
    same = _individual(1, 2)
    different = _individual(20, 30)

    assert triangle_population_diversity((first, same)) == 0.0
    assert 0.0 < triangle_population_diversity((first, different)) <= 1.0

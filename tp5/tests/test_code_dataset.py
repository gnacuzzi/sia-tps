import numpy as np

from sia_tp5 import (belongs_to_grammar, compiles, decode_patterns,
                     encode_sources, generate_function_sources,
                     load_code_dataset, semantic_signature)


def test_generator_builds_90000_unique_fixed_length_programs():
    sources = generate_function_sources()

    assert len(sources) == 90_000
    assert len(set(sources)) == 90_000
    assert {len(source) for source in sources} == {26}
    for source in sources[::997]:
        assert compiles(source)
        assert belongs_to_grammar(source)


def test_character_one_hot_encoding_round_trips_exactly():
    sources = generate_function_sources()[::13_700]
    patterns = encode_sources(sources)

    assert patterns.shape == (7, 650)
    assert set(np.unique(patterns)) == {0.0, 1.0}
    np.testing.assert_array_equal(
        patterns.reshape(7, 26, 25).sum(axis=2), np.ones((7, 26))
    )
    assert decode_patterns(patterns) == sources


def test_split_is_disjoint_complete_and_reproducible():
    first = load_code_dataset(split_seed=17)
    second = load_code_dataset(split_seed=17)

    assert len(first.train_indices) == 1_800
    assert len(first.validation_indices) == 450
    assert len(first.test_indices) == 450
    assert set(first.train_indices).isdisjoint(first.test_indices)
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert set(first.validation_indices).isdisjoint(first.test_indices)
    assert (set(first.train_indices) | set(first.validation_indices)
            | set(first.test_indices)) == set(range(2_700))
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(
        first.validation_indices, second.validation_indices
    )
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert set(first.train_sources).isdisjoint(first.test_sources)
    assert set(first.train_sources).isdisjoint(first.validation_sources)
    assert set(first.validation_sources).isdisjoint(first.test_sources)


def test_each_split_is_balanced_by_operator_pair():
    dataset = load_code_dataset()
    for sources, expected in [
        (dataset.train_sources, 200),
        (dataset.validation_sources, 50),
        (dataset.test_sources, 50),
    ]:
        counts = {(left, right): 0 for left in "+-*" for right in "+-*"}
        for source in sources:
            counts[(source[19], source[23])] += 1
        assert set(counts.values()) == {expected}


def test_semantic_signature_is_exact_for_all_operator_types():
    assert semantic_signature("def f(x): return (x+47)*83") == (83, 3901)
    assert semantic_signature("def f(x): return (x- 9)- 8") == (1, -17)
    assert semantic_signature("def f(x): return (x* 4)+ 3") == (4, 3)
    assert semantic_signature("def f(x): return (x+ 3)* 0") == (0, 0)


def test_compilation_and_grammar_are_independent_criteria():
    compilable_outside_family = "def f(x): return x**2  "
    invalid_python = "def f(x) return (x+3)*7"

    assert compiles(compilable_outside_family)
    assert not belongs_to_grammar(compilable_outside_family)
    assert not compiles(invalid_python)
    assert not belongs_to_grammar(invalid_python)


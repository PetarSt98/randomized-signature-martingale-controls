"""Exact tests for homogeneous Lyndon bases and word expansions.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_free_lie.py``).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lie_reservoir import (  # noqa: E402
    Bracket,
    Generator,
    LieNode,
    format_lie_node,
    lyndon_basis,
    lyndon_words,
    witt_dimension,
    word_expansion,
)


EXPECTED_DIMENSIONS = {
    2: (2, 1, 2, 3),
    3: (3, 3, 8, 18),
}

EXPECTED_BINARY_WORDS = {
    1: ((0,), (1,)),
    2: ((0, 1),),
    3: ((0, 0, 1), (0, 1, 1)),
    4: ((0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1)),
}

EXPECTED_TERNARY_DEGREE_THREE_WORDS = (
    (0, 0, 1),
    (0, 0, 2),
    (0, 1, 1),
    (0, 1, 2),
    (0, 2, 1),
    (0, 2, 2),
    (1, 1, 2),
    (1, 2, 2),
)


def _node_word(node: LieNode) -> tuple[int, ...]:
    if isinstance(node, Generator):
        return (node.index,)
    return _node_word(node.left) + _node_word(node.right)


def _is_lyndon_by_rotations(word: tuple[int, ...]) -> bool:
    return bool(word) and all(
        word < word[split:] + word[:split]
        for split in range(1, len(word))
    )


def _expansion_dict(node: LieNode) -> dict[tuple[int, ...], int]:
    return dict(word_expansion(node))


def _sum_expansions(*nodes: LieNode) -> dict[tuple[int, ...], int]:
    total: dict[tuple[int, ...], int] = {}
    for node in nodes:
        for word, coefficient in word_expansion(node):
            total[word] = total.get(word, 0) + coefficient
    return {word: coefficient for word, coefficient in total.items() if coefficient}


def _fraction_rank(rows: list[list[int]]) -> int:
    matrix = [[Fraction(value) for value in row] for row in rows]
    if not matrix:
        return 0

    rank = 0
    for column in range(len(matrix[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(matrix))
                if matrix[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(rank + 1, len(matrix)):
            factor = matrix[row][column]
            if factor != 0:
                matrix[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(matrix[row], matrix[rank], strict=True)
                ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _assert_raises(
    exception_type: type[Exception],
    function: Callable[..., object],
    *args: object,
) -> None:
    try:
        function(*args)
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"expected {exception_type.__name__}, got {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"expected {exception_type.__name__}")


def test_required_witt_dimension_tables() -> None:
    for input_dim, dimensions in EXPECTED_DIMENSIONS.items():
        actual = tuple(
            witt_dimension(input_dim=input_dim, degree=degree)
            for degree in range(1, 5)
        )
        assert actual == dimensions


def test_input_dim_one_has_only_the_degree_one_generator() -> None:
    assert lyndon_words(input_dim=1, degree=1) == ((0,),)
    assert lyndon_basis(input_dim=1, degree=1) == (Generator(0),)
    assert witt_dimension(input_dim=1, degree=1) == 1
    for degree in range(2, 5):
        assert lyndon_words(input_dim=1, degree=degree) == ()
        assert lyndon_basis(input_dim=1, degree=degree) == ()
        assert witt_dimension(input_dim=1, degree=degree) == 0


def test_basis_counts_equal_witt_dimensions() -> None:
    for input_dim in (2, 3):
        for degree in range(1, 5):
            expected = witt_dimension(input_dim=input_dim, degree=degree)
            assert len(lyndon_words(input_dim=input_dim, degree=degree)) == expected
            assert len(lyndon_basis(input_dim=input_dim, degree=degree)) == expected


def test_lyndon_word_order_is_deterministic() -> None:
    for degree, expected in EXPECTED_BINARY_WORDS.items():
        assert lyndon_words(input_dim=2, degree=degree) == expected
    assert (
        lyndon_words(input_dim=3, degree=3)
        == EXPECTED_TERNARY_DEGREE_THREE_WORDS
    )

    for input_dim in (2, 3):
        for degree in range(1, 5):
            words = lyndon_words(input_dim=input_dim, degree=degree)
            assert isinstance(words, tuple)
            assert words == tuple(sorted(words))
            assert words == lyndon_words(input_dim=input_dim, degree=degree)
            assert len(words) == len(set(words))
            assert all(len(word) == degree for word in words)
            assert all(_is_lyndon_by_rotations(word) for word in words)


def test_standard_factorization_and_basis_degree() -> None:
    for input_dim in (2, 3):
        for degree in range(1, 5):
            words = lyndon_words(input_dim=input_dim, degree=degree)
            basis = lyndon_basis(input_dim=input_dim, degree=degree)
            assert isinstance(basis, tuple)
            for word, node in zip(words, basis, strict=True):
                assert _node_word(node) == word
                assert len(_node_word(node)) == degree
                if degree == 1:
                    assert isinstance(node, Generator)
                    continue

                assert isinstance(node, Bracket)
                left_word = _node_word(node.left)
                right_word = _node_word(node.right)
                lyndon_suffixes = tuple(
                    word[split:]
                    for split in range(1, len(word))
                    if _is_lyndon_by_rotations(word[split:])
                )
                assert left_word + right_word == word
                assert _is_lyndon_by_rotations(left_word)
                assert _is_lyndon_by_rotations(right_word)
                assert right_word == max(lyndon_suffixes, key=len)

    words = lyndon_words(input_dim=3, degree=4)
    basis = lyndon_basis(input_dim=3, degree=4)
    node = basis[words.index((0, 1, 0, 2))]
    assert isinstance(node, Bracket)
    assert _node_word(node.left) == (0, 1)
    assert _node_word(node.right) == (0, 2)


def test_generator_and_degree_two_word_expansions() -> None:
    assert word_expansion(Generator(2)) == (((2,), 1),)
    words = lyndon_words(input_dim=3, degree=2)
    basis = lyndon_basis(input_dim=3, degree=2)
    for word, node in zip(words, basis, strict=True):
        left, right = word
        assert _expansion_dict(node) == {
            (left, right): 1,
            (right, left): -1,
        }


def test_expansions_are_exact_homogeneous_integer_vectors() -> None:
    for input_dim in (2, 3):
        for degree in range(1, 5):
            words = lyndon_words(input_dim=input_dim, degree=degree)
            basis = lyndon_basis(input_dim=input_dim, degree=degree)
            for lyndon_word, node in zip(words, basis, strict=True):
                expansion = word_expansion(node)
                expansion_dict = dict(expansion)
                assert expansion == tuple(sorted(expansion))
                assert len(expansion) == len(expansion_dict)
                assert expansion_dict[lyndon_word] == 1
                assert all(len(word) == degree for word in expansion_dict)
                assert all(
                    type(coefficient) is int and coefficient != 0
                    for coefficient in expansion_dict.values()
                )


def test_exact_antisymmetry_at_word_level() -> None:
    first = Bracket(Generator(0), Generator(1))
    second = Bracket(Generator(1), Generator(2))
    assert _sum_expansions(
        Bracket(first, second),
        Bracket(second, first),
    ) == {}


def test_exact_jacobi_combination_expands_to_zero() -> None:
    first = Generator(0)
    second = Generator(1)
    third = Generator(2)
    assert _sum_expansions(
        Bracket(first, Bracket(second, third)),
        Bracket(second, Bracket(third, first)),
        Bracket(third, Bracket(first, second)),
    ) == {}


def test_basis_expansions_are_rationally_independent() -> None:
    assert _fraction_rank([[1, 2], [2, 4]]) == 1
    for input_dim in (2, 3):
        for degree in range(1, 5):
            associative_words = tuple(product(range(input_dim), repeat=degree))
            rows = [
                [
                    _expansion_dict(node).get(word, 0)
                    for word in associative_words
                ]
                for node in lyndon_basis(input_dim=input_dim, degree=degree)
            ]
            assert _fraction_rank(rows) == witt_dimension(
                input_dim=input_dim,
                degree=degree,
            )


def test_human_formatting_is_one_based_and_deterministic() -> None:
    node = Bracket(
        Generator(0),
        Bracket(Generator(1), Generator(2)),
    )
    assert format_lie_node(Generator(0)) == "X1"
    assert format_lie_node(node) == "[X1,[X2,X3]]"

    words = lyndon_words(input_dim=3, degree=4)
    basis = lyndon_basis(input_dim=3, degree=4)
    factorization_regression = basis[words.index((0, 1, 0, 2))]
    assert format_lie_node(factorization_regression) == "[[X1,X2],[X1,X3]]"


def test_invalid_arguments_are_rejected_clearly() -> None:
    for function in (lyndon_words, lyndon_basis, witt_dimension):
        _assert_raises(ValueError, function, 0, 1)
        _assert_raises(ValueError, function, 2, 0)
        _assert_raises(TypeError, function, True, 1)
        _assert_raises(TypeError, function, 2, 1.5)

    _assert_raises(TypeError, Generator, True)
    _assert_raises(ValueError, Generator, -1)
    _assert_raises(TypeError, Bracket, Generator(0), "not a Lie node")
    _assert_raises(TypeError, word_expansion, "not a Lie node")
    _assert_raises(TypeError, format_lie_node, "not a Lie node")


def _run_all() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()

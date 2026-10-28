"""Exact regression checks for the finite tensor-space witness harness."""

from __future__ import annotations

import ast
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

import lie_reservoir.tensor_witness as witness
from lie_reservoir.finite_observation import exact_matrix_rank


F = Fraction
ROOT = Path(__file__).resolve().parents[1]
CASES = witness.STRUCTURAL_CASES


def _integer_identity(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(row == column) for column in range(size))
        for row in range(size)
    )


def _fraction_identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(F(int(row == column)) for column in range(size))
        for row in range(size)
    )


def _basis_vector(index: int, size: int) -> tuple[int, ...]:
    return tuple(int(coordinate == index) for coordinate in range(size))


def _matrix_vector_product(
    matrix: tuple[tuple[int, ...], ...],
    vector: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        sum(entry * value for entry, value in zip(row, vector, strict=True))
        for row in matrix
    )


def test_word_order_is_degree_first_then_lexicographic() -> None:
    assert witness.tensor_word_basis(2, 2) == (
        (),
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    for input_dim, degree in CASES:
        basis = witness.tensor_word_basis(input_dim, degree)
        expected = ((),) + tuple(
            word
            for level in range(1, degree + 1)
            for word in product(range(input_dim), repeat=level)
        )
        assert basis == expected
        assert len(basis) == witness.tensor_state_dimension(input_dim, degree)
        assert basis[1:] == witness.nonempty_tensor_words(input_dim, degree)


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_right_concatenation_action_is_exact(
    input_dim: int,
    degree: int,
) -> None:
    basis = witness.tensor_word_basis(input_dim, degree)
    indices = {word: index for index, word in enumerate(basis)}
    actions = witness.right_concatenation_actions(input_dim, degree)
    zero = (0,) * len(basis)
    for letter, action in enumerate(actions):
        for word, source in indices.items():
            actual = witness.apply_right_action(
                action,
                _basis_vector(source, len(basis)),
            )
            if len(word) == degree:
                assert actual == zero
            else:
                assert actual == _basis_vector(
                    indices[word + (letter,)],
                    len(basis),
                )


def test_left_concatenation_mutation_fails_chronological_identity() -> None:
    input_dim, degree = 2, 2
    basis = witness.tensor_word_basis(input_dim, degree)
    indices = {word: index for index, word in enumerate(basis)}
    left_actions = tuple(
        tuple(
            indices[(letter,) + word] if len(word) < degree else None
            for word in basis
        )
        for letter in range(input_dim)
    )
    word = (0, 1)
    mutated = witness.gamma_linear_word_evaluation(word, left_actions)
    expected = _basis_vector(indices[word], len(basis))
    assert mutated == _basis_vector(indices[(1, 0)], len(basis))
    assert mutated != expected


def test_operator_order_mutation_fails_chronological_identity() -> None:
    input_dim, degree = 2, 2
    basis = witness.tensor_word_basis(input_dim, degree)
    indices = {word: index for index, word in enumerate(basis)}
    actions = witness.right_concatenation_actions(input_dim, degree)
    word = (0, 1)
    state = _basis_vector(0, len(basis))
    for letter in reversed(word):
        state = witness.apply_right_action(actions[letter], state)
    expected = _basis_vector(indices[word], len(basis))
    assert state == _basis_vector(indices[(1, 0)], len(basis))
    assert state != expected
    assert witness.gamma_linear_word_evaluation(word, actions) == expected


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_complete_word_matrix_is_exact_identity(
    input_dim: int,
    degree: int,
) -> None:
    word_count = len(witness.nonempty_tensor_words(input_dim, degree))
    matrix = witness.word_evaluation_matrix(input_dim, degree)
    assert matrix == _integer_identity(word_count)
    assert all(type(value) is int for row in matrix for value in row)


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_length_degree_plus_one_cutoff_is_exact(
    input_dim: int,
    degree: int,
) -> None:
    assert witness.length_cutoff_holds(input_dim, degree)


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_one_segment_signature_equals_linear_system_response(
    input_dim: int,
    degree: int,
) -> None:
    first, _ = witness.deterministic_segment_increments(input_dim, degree)
    actions = witness.right_concatenation_actions(input_dim, degree)
    expected = witness.segment_signature(first, input_dim, degree)
    actual = witness.linear_segment_response(first, degree, actions)
    assert actual == expected
    assert all(isinstance(value, Fraction) for value in actual)


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_two_segment_chen_product_equals_successive_response(
    input_dim: int,
    degree: int,
) -> None:
    first, second = witness.deterministic_segment_increments(input_dim, degree)
    actions = witness.right_concatenation_actions(input_dim, degree)
    first_tensor = witness._segment_signature_tensor(first, input_dim, degree)
    second_tensor = witness._segment_signature_tensor(second, input_dim, degree)
    chen_tensor = witness._exact_chen_product(
        first_tensor,
        second_tensor,
        input_dim,
        degree,
    )
    expected = tuple(
        chen_tensor.get(word, F(0))
        for word in witness.tensor_word_basis(input_dim, degree)
    )
    first_response = witness.linear_segment_response(first, degree, actions)
    actual = witness.linear_segment_response(
        second,
        degree,
        actions,
        initial_state=first_response,
    )
    assert actual == expected
    assert witness.signature_convention_checks(input_dim, degree) == (True, True)


def test_ternary_chen_cross_term_has_chronological_order() -> None:
    input_dim, degree = 3, 2
    first, second = witness.deterministic_segment_increments(input_dim, degree)
    left = witness._segment_signature_tensor(first, input_dim, degree)
    right = witness._segment_signature_tensor(second, input_dim, degree)
    chen = witness._exact_chen_product(left, right, input_dim, degree)
    assert chen[(0, 2)] == (
        first[0] * first[2] / 2
        + first[0] * second[2]
        + second[0] * second[2] / 2
    )
    assert chen[(2, 0)] == (
        first[2] * first[0] / 2
        + first[2] * second[0]
        + second[2] * second[0] / 2
    )


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_hall_expansion_matches_exact_linear_lie_representation(
    input_dim: int,
    degree: int,
) -> None:
    actions = witness.right_concatenation_actions(input_dim, degree)
    nodes = witness.hall_basis_through_degree(input_dim, degree)
    columns = witness.hall_word_column_matrix(input_dim, degree)
    x0 = _basis_vector(0, witness.tensor_state_dimension(input_dim, degree))
    for column, node in enumerate(nodes):
        represented = witness.represented_linear_lie_matrix(node, actions)
        expanded = witness.expanded_linear_lie_matrix(node, actions)
        assert represented == expanded
        evaluated = _matrix_vector_product(represented, x0)[1:]
        expected = tuple(int(row[column]) for row in columns)
        assert evaluated == expected


@pytest.mark.parametrize(("input_dim", "degree"), CASES)
def test_hall_column_matrix_has_exact_full_column_rank(
    input_dim: int,
    degree: int,
) -> None:
    matrix = witness.hall_word_column_matrix(input_dim, degree)
    column_count = len(witness.hall_basis_through_degree(input_dim, degree))
    assert exact_matrix_rank(matrix) == column_count
    assert witness.hall_convention_checks(input_dim, degree) == (
        True,
        column_count,
        column_count,
    )


@pytest.mark.parametrize(("input_dim", "degree"), witness.BLOCK_EMBEDDING_CASES)
def test_inactive_coordinate_block_embedding_is_exact(
    input_dim: int,
    degree: int,
) -> None:
    assert witness.block_embedding_checks(
        input_dim,
        degree,
        extra_dimensions=2,
    ) == (True, True, True)


@pytest.mark.parametrize(
    ("input_dim", "degree"),
    witness.TANH_LIMIT_DIAGNOSTIC_CASES,
)
def test_normalized_tanh_limit_constant_is_identity_and_linear_term_is_zero(
    input_dim: int,
    degree: int,
) -> None:
    pytest.importorskip("sympy")
    constant, linear = (
        witness.noncertifying_normalized_tanh_limit_coefficients(
            input_dim,
            degree,
        )
    )
    word_count = len(witness.nonempty_tensor_words(input_dim, degree))
    assert constant == _fraction_identity(word_count)
    assert linear == tuple(
        tuple(F(0) for _ in range(word_count))
        for _ in range(word_count)
    )
    assert all(
        isinstance(value, Fraction)
        for matrix in (constant, linear)
        for row in matrix
        for value in row
    )


def test_tanh_limit_diagnostic_is_explicitly_noncertifying() -> None:
    assert not witness.NORMALIZED_TANH_LIMIT_CERTIFIES_NONZERO_EPSILON
    assert "noncertifying" in (
        witness.noncertifying_normalized_tanh_limit_coefficients.__name__
    )
    assert "never evaluates a nonzero epsilon" in (
        witness.noncertifying_normalized_tanh_limit_coefficients.__doc__ or ""
    )


def test_certifying_module_has_no_floating_rank_or_determinant_path() -> None:
    source = Path(witness.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_modules.update(alias.name for alias in node.names)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imported_modules)
    assert {
        "float",
        "det",
        "determinant",
        "matrix_rank",
        "rank",
    }.isdisjoint(called_names)
    assert "_determinant" not in imported_modules
    assert {"det", "determinant", "matrix_rank", "rank"}.isdisjoint(
        called_attributes
    )
    assert not any(
        isinstance(node, ast.Constant) and isinstance(node.value, float)
        for node in ast.walk(tree)
    )


def test_input_two_degree_three_complete_exact_chain() -> None:
    input_dim, degree = 2, 3
    basis = witness.tensor_word_basis(input_dim, degree)
    actions = witness.right_concatenation_actions(input_dim, degree)
    assert len(basis) == 15
    assert len(actions) == input_dim
    assert witness.word_evaluation_matrix(input_dim, degree) == _integer_identity(14)
    assert witness.signature_convention_checks(input_dim, degree) == (True, True)
    hall_equal, rank, columns = witness.hall_convention_checks(input_dim, degree)
    assert hall_equal
    assert rank == columns == 5
    assert witness.block_embedding_checks(input_dim, degree) == (True, True, True)


def test_reproducer_runs_all_cases_without_writing(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research_output" / "reproduce_tensor_witness.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == len(CASES)
    assert all("A_0=identity" in line for line in lines)
    assert all("signature=pass(one,two)" in line for line in lines)
    assert all("Hall-rank=pass" in line for line in lines)
    assert all("non-certifying" in line for line in lines)
    assert tuple(tmp_path.iterdir()) == before

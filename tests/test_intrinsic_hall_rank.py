"""Exact checks of the grading witness and its truncated observation map."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import json
import sys

import pytest
import sympy as sp

from lie_reservoir import intrinsic_hall_rank as intrinsic
from lie_reservoir.free_lie import Bracket, Generator, lyndon_basis, witt_dimension, word_expansion
from lie_reservoir.graded_reconstruction import tensor_exp


REFERENCE_PAIRS = ((2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4))


@lru_cache(maxsize=None)
def _evaluated(d: int, degree: int, state_dim: int | None = None):
    witness = intrinsic.grading_witness(d, degree, state_dim)
    return witness, intrinsic.evaluate_hall_matrix(witness)


def _assert_exact_matrix(matrix, rows: int, columns: int) -> None:
    assert len(matrix) == rows
    assert all(len(row) == columns for row in matrix)
    assert all(isinstance(value, F) for row in matrix for value in row)


@pytest.mark.parametrize(("d", "degree"), REFERENCE_PAIRS)
def test_reference_witnesses_have_exact_maximal_hall_rank(d, degree):
    witness, matrix = _evaluated(d, degree)
    dimensions = tuple(witt_dimension(d, m) for m in range(1, degree + 1))
    basis = tuple(h for m in range(1, degree + 1) for h in lyndon_basis(d, m))
    degrees = tuple(m for m, size in enumerate(dimensions, 1) for _ in range(size))
    assert witness.input_dim == d
    assert witness.degree == degree
    assert witness.homogeneous_dimensions == dimensions
    assert witness.nu == sum(dimensions)
    assert witness.state_dim == witness.nu
    assert witness.basis == basis
    assert witness.hall_degrees == degrees
    assert witness.retained_columns == tuple(range(witness.nu))
    assert witness.state_degrees == degrees
    assert witness.grading == tuple(map(F, degrees))
    _assert_exact_matrix(matrix, witness.nu, witness.nu)

    for row, row_degree in enumerate(degrees):
        for column, column_degree in enumerate(degrees):
            if row_degree > column_degree:
                assert matrix[row][column] == 0
            elif row_degree == column_degree:
                assert matrix[row][column] == F(row == column, 2)
    minor = intrinsic.witness_minor(witness, matrix)
    assert sp.Matrix(minor).det() == sp.Rational(1, 2**witness.nu)
    assert sp.Matrix(matrix).rank() == witness.nu


def test_degree_four_matrix_contains_the_two_nonidentity_entries():
    witness, matrix = _evaluated(2, 4)
    expected = [[F(row == column, 2) for column in range(8)] for row in range(8)]
    expected[2][5] = expected[2][7] = F(-1, 32)
    assert matrix == tuple(map(tuple, expected))
    assert witness.hall_degrees == (1, 1, 2, 3, 3, 4, 4, 4)
    assert sp.Matrix(matrix).det() == sp.Rational(1, 256)


@pytest.mark.parametrize("state_dim", (1, 4, 7, 9))
def test_graded_quotients_and_inactive_embedding(state_dim):
    witness, matrix = _evaluated(2, 4, state_dim)
    _, full = _evaluated(2, 4)
    active = min(state_dim, 8)
    expected = full[:active] + ((F(0),) * 8,) * (state_dim - active)
    assert matrix == expected
    assert witness.retained_columns == tuple(range(active))
    assert witness.state_degrees == (1, 1, 2, 3, 3, 4, 4, 4)[:active] + (0,) * (
        state_dim - active
    )
    _assert_exact_matrix(matrix, state_dim, 8)
    assert sp.Matrix(matrix).rank() == active
    assert sp.Matrix(intrinsic.witness_minor(witness, matrix)).det() == sp.Rational(
        1, 2**active
    )
    if state_dim in (4, 7):
        # A discarded Hall basis vector can still give a lower-grade value.
        assert matrix[2][7] == F(-1, 32)


def test_state_dimension_below_generator_count_uses_the_quotient():
    witness, matrix = _evaluated(3, 3, 2)
    assert witness.retained_columns == (0, 1)
    assert witness.projected_generators == ((F(1), F(0)), (F(0), F(1)), (F(0), F(0)))
    assert witness.bias_q_multipliers == witness.projected_generators
    assert all(value == 0 for adjoint in witness.adjoints for row in adjoint for value in row)
    expected = tuple(tuple(F(row == column, 2) for column in range(14)) for row in range(2))
    assert matrix == expected


@pytest.mark.parametrize("degree", (1, 5))
def test_one_generator_edge(degree):
    witness, matrix = _evaluated(1, degree)
    assert witness.nu == 1
    assert witness.homogeneous_dimensions == (1,) + (0,) * (degree - 1)
    assert matrix == ((F(1, 2),),)
    assert intrinsic.exact_tanh_jets(witness) == (({(): F(1, 2)},),)


def test_parameters_follow_the_graded_adjoint_and_tanh_jet_conventions():
    witness = intrinsic.grading_witness(2, 4, 9)
    for adjoint, lam, weight in zip(witness.adjoints, witness.lambdas, witness.weights):
        _assert_exact_matrix(adjoint, 9, 9)
        _assert_exact_matrix(lam, 9, 9)
        for row in range(9):
            for column in range(9):
                expected = (
                    adjoint[row][column] * witness.grading[column] / witness.grading[row]
                    if witness.grading[row]
                    else F(0)
                )
                assert lam[row][column] == expected
                assert weight[row][column] == -expected
                if row == 8 or column == 8:
                    assert adjoint[row][column] == 0
    jets = intrinsic.exact_tanh_jets(witness)
    assert jets[0][0] == {(): F(1, 2)}
    assert jets[1][1] == {(): F(1, 2)}
    assert jets[0][2] == {(1,): F(-1, 2), (1, 1, 1): F(1, 24)}
    assert jets[1][2] == {(0,): F(1, 2), (0, 0, 0): F(-1, 24)}
    assert jets[0][8] == jets[1][8] == {}
    for vector in jets:
        assert len(vector) == 9
        for polynomial in vector:
            for monomial, coefficient in polynomial.items():
                assert monomial == tuple(sorted(monomial))
                assert len(monomial) <= witness.degree - 1
                assert all(0 <= variable < 9 for variable in monomial)
                assert isinstance(coefficient, F)
                assert coefficient != 0


def test_affine_representation_and_cocycle_on_all_degree_four_hall_pairs():
    witness = intrinsic.grading_witness(2, 4)
    adjoints = tuple(map(sp.Matrix, witness.adjoints))
    lambdas = tuple(map(sp.Matrix, witness.lambdas))
    coordinates = sp.eye(witness.nu)

    @lru_cache(maxsize=None)
    def represented(node, use_lambda):
        if isinstance(node, Generator):
            return (lambdas if use_lambda else adjoints)[node.index]
        left = represented(node.left, use_lambda)
        right = represented(node.right, use_lambda)
        return left * right - right * left

    for a_index, a in enumerate(witness.basis):
        lambda_a = represented(a, True)
        for b_index, b in enumerate(witness.basis):
            lambda_b = represented(b, True)
            bracket_coordinates = represented(a, False) * coordinates[:, b_index]
            cocycle = lambda_a * coordinates[:, b_index] - lambda_b * coordinates[:, a_index]
            assert cocycle == bracket_coordinates

            # Check the entire associative expansion, beyond pivot words.
            actual_tensor = {}
            for coefficient, hall in zip(bracket_coordinates, witness.basis):
                for word, factor in word_expansion(hall):
                    actual_tensor[word] = actual_tensor.get(word, 0) + coefficient * factor
            actual_tensor = {word: value for word, value in actual_tensor.items() if value}
            expected_tensor = {
                word: coefficient
                for word, coefficient in word_expansion(Bracket(a, b))
                if len(word) <= witness.degree
            }
            assert actual_tensor == expected_tensor
            lambda_bracket = sp.zeros(witness.nu)
            for coefficient, hall in zip(bracket_coordinates, witness.basis):
                lambda_bracket += coefficient * represented(hall, True)
            assert lambda_a * lambda_b - lambda_b * lambda_a == lambda_bracket


def test_tanh_coefficients_match_independent_symbolic_derivatives():
    t = sp.Symbol("t")
    expected = (F(0), F(1), F(0), F(-1, 3), F(0), F(2, 15), F(0), F(-17, 315))
    assert intrinsic.tanh_coefficients(F(0), 7) == expected
    for value, order in ((F(0), 7), (F(1, 2), 4)):
        point = sp.atanh(sp.Rational(value.numerator, value.denominator))
        coefficients = intrinsic.tanh_coefficients(value, order)
        assert len(coefficients) == order + 1
        for k, coefficient in enumerate(coefficients):
            direct = sp.simplify(sp.diff(sp.tanh(t), t, k).subs(t, point) / sp.factorial(k))
            assert coefficient == direct


def test_hall_evaluation_uses_actual_field_parameters():
    witness = intrinsic.grading_witness(2, 4)
    zeros = tuple((F(0),) * witness.state_dim for _ in range(witness.state_dim))
    constant_fields = replace(witness, weights=(zeros,) * witness.input_dim)
    matrix = intrinsic.evaluate_hall_matrix(constant_fields)
    expected = tuple(
        tuple(F(row == column and column < 2, 2) for column in range(witness.nu))
        for row in range(witness.state_dim)
    )
    assert matrix == expected


def _direct_degree_three_fields():
    """Actual tanh fields, written explicitly independently of the new module."""
    x = sp.symbols("x0:5")
    c = sp.Rational(1, 2)
    fields = (
        sp.Matrix([c, 0, -sp.tanh(x[1] / 2), -sp.tanh(2 * x[2] / 3), 0]),
        sp.Matrix([0, c, sp.tanh(x[0] / 2), 0, sp.tanh(2 * x[2] / 3)]),
    )
    return x, fields


def test_bracket_signs_by_direct_actual_tanh_differentiation():
    x, fields = _direct_degree_three_fields()
    at_zero = dict.fromkeys(x, 0)
    first, second = fields
    bracket = second.jacobian(x) * first - first.jacobian(x) * second
    nested_left = bracket.jacobian(x) * first - first.jacobian(x) * bracket
    nested_right = second.jacobian(x) * bracket - bracket.jacobian(x) * second
    direct = sp.Matrix.hstack(first, second, bracket, nested_left, nested_right).subs(at_zero)
    assert direct == sp.eye(5) / 2
    _, matrix = _evaluated(2, 3)
    assert sp.Matrix(matrix) == direct


def test_full_truncated_polynomial_and_zero_jacobian_independently():
    """Interpolate the full weighted polynomial from existing tensor_exp values.

    Each response uses direct derivatives of actual tanh fields.  This checks
    nonlinear terms and tensor factorials before taking the zero Jacobian.
    """
    witness = intrinsic.grading_witness(2, 3)
    x, fields = _direct_degree_three_fields()
    at_zero = dict.fromkeys(x, 0)
    z = sp.symbols("z0:5")
    scale = sp.Symbol("r")
    degrees = (1, 1, 2, 3, 3)
    exponents = tuple(
        powers
        for powers in product(range(4), repeat=5)
        if sum(m * k for m, k in zip(degrees, powers)) <= 3
    )
    assert len(exponents) == 15

    @lru_cache(maxsize=None)
    def direct_word(word):
        if len(word) == 1:
            return fields[word[0]]
        # The first letter is the outermost directional derivative.
        return direct_word(word[1:]).jacobian(x) * fields[word[0]]

    values = {
        word: direct_word(word).subs(at_zero)
        for length in range(1, 4)
        for word in product(range(2), repeat=length)
    }
    basis = tuple(h for m in range(1, 4) for h in lyndon_basis(2, m))
    responses = []
    for point in exponents:
        lie_tensor = {}
        for coordinate, hall in zip(point, basis):
            for word, coefficient in word_expansion(hall):
                lie_tensor[word] = lie_tensor.get(word, F(0)) + F(coordinate * coefficient)
        signature = tensor_exp(lie_tensor)
        response = sp.zeros(5, 1)
        for word, coefficient in signature.items():
            if 1 <= len(word) <= 3:
                response += coefficient * scale ** len(word) * values[word]
        responses.append(list(response))

    # The integer nodes indexed by this lower set are unisolvent.
    vandermonde = sp.Matrix(
        [[sp.prod(v**p for v, p in zip(point, powers)) for powers in exponents] for point in exponents]
    )
    coefficients = vandermonde.inv() * sp.Matrix(responses)
    monomials = sp.Matrix([[sp.prod(v**p for v, p in zip(z, powers)) for powers in exponents]])
    full_map = (monomials * coefficients).T.applyfunc(sp.expand)
    expected = sp.Matrix(
        [
            scale * z[0] / 2,
            scale * z[1] / 2,
            scale**2 * z[2] / 2,
            scale**3 * (z[3] / 2 - z[0] * z[2] / 12),
            scale**3 * (z[4] / 2 + z[1] * z[2] / 12),
        ]
    )
    assert (full_map - expected).applyfunc(sp.expand) == sp.zeros(5, 1)
    zero_jacobian = full_map.jacobian(z).subs(dict.fromkeys(z, 0))
    for rational_scale in (F(0), F(1), F(1, 3), F(-2)):
        actual = intrinsic.zero_response_jacobian(witness, rational_scale)
        _assert_exact_matrix(actual, 5, 5)
        assert sp.Matrix(actual) == zero_jacobian.subs(scale, rational_scale)


@pytest.mark.parametrize("state_dim", (4, 9))
@pytest.mark.parametrize("scale", (F(0), F(1, 3), F(-2)))
def test_rectangular_zero_response_jacobian_scales_each_hall_column(state_dim, scale):
    witness, matrix = _evaluated(2, 4, state_dim)
    actual = intrinsic.zero_response_jacobian(witness, scale)
    expected = tuple(
        tuple(value * scale**degree for value, degree in zip(row, witness.hall_degrees))
        for row in matrix
    )
    assert actual == expected


@pytest.mark.parametrize("arguments", ((0, 2), (2, 0), (2, 2, 0), (4, 2), (2, 6), (2, 2, 41), (3, 5)))
def test_witness_rejects_values_outside_the_documented_bounds(arguments):
    with pytest.raises(ValueError):
        intrinsic.grading_witness(*arguments)


@pytest.mark.parametrize("arguments", ((True, 2), (2, True), (2, 2, True), (2.0, 2)))
def test_witness_rejects_noninteger_dimensions(arguments):
    with pytest.raises(TypeError):
        intrinsic.grading_witness(*arguments)


def test_default_certificate_verifier_is_read_only_and_rejects_entry_tampering(
    tmp_path, monkeypatch, capsys
):
    from research_output import reproduce_intrinsic_hall_rank as reproduce

    original = reproduce.ARTIFACT
    original_bytes = original.read_bytes()
    original_mtime = original.stat().st_mtime_ns
    temporary = tmp_path / original.name
    temporary.write_bytes(original_bytes)
    monkeypatch.setattr(reproduce, "ARTIFACT", temporary)
    monkeypatch.setattr(sys, "argv", [str(reproduce.__file__)])
    temporary_mtime = temporary.stat().st_mtime_ns
    assert reproduce.main() == 0
    assert "verified (17 exact finite cases)" in capsys.readouterr().out
    assert temporary.read_bytes() == original_bytes
    assert temporary.stat().st_mtime_ns == temporary_mtime

    tampered = json.loads(original_bytes)
    full_eight = next(
        case
        for case in tampered["cases"]
        if (case["input_dimension"], case["maximum_degree"], case["state_dimension"])
        == (2, 4, 8)
    )
    # Preserve the rank, determinant, triangular pattern, and minor slice.
    # A verifier that checked only these summaries would miss this change.
    full_eight["exact_hall_evaluation_matrix"][2][5] = "0"
    full_eight["exact_selected_minor"][2][5] = "0"
    changed_matrix = sp.Matrix(full_eight["exact_hall_evaluation_matrix"])
    assert changed_matrix.det() == sp.Rational(1, 256)
    assert changed_matrix.rank() == 8
    tampered_bytes = (json.dumps(tampered, indent=2, sort_keys=True) + "\n").encode()
    temporary.write_bytes(tampered_bytes)
    tampered_mtime = temporary.stat().st_mtime_ns
    with pytest.raises(RuntimeError, match="stale or inconsistent"):
        reproduce.main()
    assert temporary.read_bytes() == tampered_bytes
    assert temporary.stat().st_mtime_ns == tampered_mtime
    assert original.read_bytes() == original_bytes
    assert original.stat().st_mtime_ns == original_mtime

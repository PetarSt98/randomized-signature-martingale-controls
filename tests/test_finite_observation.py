from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import root

from lie_reservoir.certificates import (
    _degree_data,
    _determinant,
    _differentiate_series,
    _multiply_series,
    _tanh_jet,
)
from lie_reservoir.finite_observation import (
    candidate_state_scale_pairs,
    certify_exact_design,
    common_rescale_design,
    exact_centered_truncated_observation_map,
    exact_joint_state_scale_matrix,
    exact_matrix_rank,
    exact_truncated_observation_map,
    exact_zero_jacobian,
    exact_zero_jacobian_from_tensor,
    hall_basis_degrees,
    hall_basis_labels,
    noncertifying_spectral_diagnostics,
    primary_candidate_state_scale_pairs,
    select_exact_design,
    single_scale_field_jet_matrix,
)
from lie_reservoir.free_lie import lyndon_basis, word_expansion


F = Fraction
ROOT = Path(__file__).resolve().parents[1]

ZERO_COORDINATES = (
    (F(0), F(0)),
    (F(0),),
    (F(0), F(0)),
    (F(0), F(0), F(0)),
)

EXPECTED_LABELS = (
    "X1",
    "X2",
    "[X1,X2]",
    "[X1,[X1,X2]]",
    "[[X1,X2],X2]",
    "[X1,[X1,[X1,X2]]]",
    "[X1,[[X1,X2],X2]]",
    "[[[X1,X2],X2],X2]",
)

EXPECTED_DEGREES = (1, 1, 2, 3, 3, 4, 4, 4)

EXPECTED_DESIGN = (
    ("-log(3)/2", F(1)),
    ("-log(3)/2", F(2)),
    ("-log(3)/2", F(3)),
    ("0", F(2)),
    ("0", F(3)),
    ("log(3)/2", F(1)),
    ("log(3)/2", F(2)),
    ("log(3)/2", F(3)),
)

EXPECTED_MATRIX = (
    (
        F(-1, 2),
        F(1, 2),
        F(-3, 8),
        F(15, 32),
        F(3, 16),
        F(-57, 128),
        F(-3, 32),
        F(3, 64),
    ),
    (
        F(-1),
        F(1),
        F(-3, 2),
        F(15, 4),
        F(3, 2),
        F(-57, 8),
        F(-3, 2),
        F(3, 4),
    ),
    (
        F(-3, 2),
        F(3, 2),
        F(-27, 8),
        F(405, 32),
        F(81, 16),
        F(-4617, 128),
        F(-243, 32),
        F(243, 64),
    ),
    (F(0), F(1), F(-2), F(4), F(0), F(-8), F(0), F(4)),
    (F(0), F(3, 2), F(-9, 2), F(27, 2), F(0), F(-81, 2), F(0), F(81, 4)),
    (
        F(1, 2),
        F(1, 2),
        F(-3, 8),
        F(15, 32),
        F(-3, 16),
        F(-57, 128),
        F(3, 32),
        F(3, 64),
    ),
    (
        F(1),
        F(1),
        F(-3, 2),
        F(15, 4),
        F(-3, 2),
        F(-57, 8),
        F(3, 2),
        F(3, 4),
    ),
    (
        F(3, 2),
        F(3, 2),
        F(-27, 8),
        F(405, 32),
        F(-81, 16),
        F(-4617, 128),
        F(243, 32),
        F(243, 64),
    ),
)

EXPECTED_DETERMINANT = F(-258733035, 524288)

EXPECTED_INVERSE = (
    (
        F(-18, 11),
        F(9, 22),
        F(-2, 33),
        F(0),
        F(0),
        F(18, 11),
        F(-9, 22),
        F(2, 33),
    ),
    (
        F(360, 239),
        F(126, 239),
        F(-88, 717),
        F(-405, 239),
        F(80, 239),
        F(360, 239),
        F(126, 239),
        F(-88, 717),
    ),
    (
        F(-76, 239),
        F(547, 239),
        F(-964, 2151),
        F(-990, 239),
        F(1760, 2151),
        F(-76, 239),
        F(547, 239),
        F(-964, 2151),
    ),
    (
        F(-760, 717),
        F(1168, 717),
        F(-664, 2151),
        F(-432, 239),
        F(256, 717),
        F(-760, 717),
        F(1168, 717),
        F(-664, 2151),
    ),
    (
        F(-76, 33),
        F(52, 33),
        F(-28, 99),
        F(0),
        F(0),
        F(76, 33),
        F(-52, 33),
        F(28, 99),
    ),
    (
        F(-3056, 10755),
        F(476, 1195),
        F(-2992, 32265),
        F(-1244, 3585),
        F(808, 10755),
        F(-3056, 10755),
        F(476, 1195),
        F(-2992, 32265),
    ),
    (
        F(-40, 33),
        F(32, 33),
        F(-8, 33),
        F(0),
        F(0),
        F(40, 33),
        F(-32, 33),
        F(8, 33),
    ),
    (
        F(-472, 10755),
        F(646, 3585),
        F(-2264, 32265),
        F(-1018, 3585),
        F(1276, 10755),
        F(-472, 10755),
        F(646, 3585),
        F(-2264, 32265),
    ),
)


def _independent_hall_tensor(
    flat_coordinates: tuple[Fraction, ...],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    coordinate_index = 0
    for degree in range(1, 5):
        for node in lyndon_basis(input_dim=2, degree=degree):
            coefficient = flat_coordinates[coordinate_index]
            coordinate_index += 1
            for word, integer_coefficient in word_expansion(node):
                result[word] = (
                    result.get(word, F(0))
                    + coefficient * integer_coefficient
                )
    return {word: value for word, value in result.items() if value != 0}


def _independent_tensor_product(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            word = left_word + right_word
            if len(word) <= 4:
                result[word] = (
                    result.get(word, F(0)) + left_value * right_value
                )
    return {word: value for word, value in result.items() if value != 0}


def _independent_tensor_exp(
    lie_element: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result = {(): F(1)}
    power = {(): F(1)}
    for exponent in range(1, 5):
        power = _independent_tensor_product(power, lie_element)
        for word, coefficient in power.items():
            result[word] = (
                result.get(word, F(0))
                + coefficient / F(math.factorial(exponent))
            )
    return {word: value for word, value in result.items() if value != 0}


def _independent_word_value(
    word: tuple[int, ...],
    activation_values: tuple[Fraction, Fraction],
) -> Fraction:
    slopes = (F(1), F(0))
    order = len(word) - 1
    current = _tanh_jet(
        activation_values[word[-1]],
        slopes[word[-1]],
        order,
    )
    for letter in reversed(word[:-1]):
        order -= 1
        current = _multiply_series(
            _differentiate_series(current, order),
            _tanh_jet(activation_values[letter], slopes[letter], order),
            order,
        )
    return current[0]


def _state_values() -> dict[str, tuple[Fraction, Fraction]]:
    _, labels, values, _, _ = _degree_data(4)
    return dict(zip(labels, values, strict=True))


def _independent_exact_observation(
    flat_coordinates: tuple[Fraction, ...],
    design=EXPECTED_DESIGN,
) -> tuple[Fraction, ...]:
    signature = _independent_tensor_exp(
        _independent_hall_tensor(flat_coordinates)
    )
    values = _state_values()
    return tuple(
        sum(
            (
                coefficient
                * scale ** len(word)
                * _independent_word_value(word, values[state_label])
                for word, coefficient in signature.items()
                if word
            ),
            start=F(0),
        )
        for state_label, scale in design
    )


def _numeric_tensor_product(
    left: dict[tuple[int, ...], float],
    right: dict[tuple[int, ...], float],
) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            word = left_word + right_word
            if len(word) <= 4:
                result[word] = result.get(word, 0.0) + left_value * right_value
    return result


def _independent_numeric_observation(
    coordinates: np.ndarray,
    design=EXPECTED_DESIGN,
) -> np.ndarray:
    lie_element: dict[tuple[int, ...], float] = {}
    coordinate_index = 0
    for degree in range(1, 5):
        for node in lyndon_basis(input_dim=2, degree=degree):
            coefficient = float(coordinates[coordinate_index])
            coordinate_index += 1
            for word, integer_coefficient in word_expansion(node):
                lie_element[word] = (
                    lie_element.get(word, 0.0)
                    + coefficient * integer_coefficient
                )
    signature = {(): 1.0}
    power = {(): 1.0}
    for exponent in range(1, 5):
        power = _numeric_tensor_product(power, lie_element)
        for word, coefficient in power.items():
            signature[word] = (
                signature.get(word, 0.0)
                + coefficient / math.factorial(exponent)
            )
    values = _state_values()
    return np.asarray(
        [
            sum(
                coefficient
                * float(scale) ** len(word)
                * float(_independent_word_value(word, values[state_label]))
                for word, coefficient in signature.items()
                if word
            )
            for state_label, scale in design
        ],
        dtype=float,
    )


def _blocks(flat_coordinates: tuple[Fraction, ...]):
    return (
        flat_coordinates[0:2],
        flat_coordinates[2:3],
        flat_coordinates[3:5],
        flat_coordinates[5:8],
    )


def _matrix_product(left, right):
    return tuple(
        tuple(
            sum(
                (left[row][index] * right[index][column] for index in range(8)),
                start=F(0),
            )
            for column in range(8)
        )
        for row in range(8)
    )


def _identity():
    return tuple(
        tuple(F(int(row == column)) for column in range(8))
        for row in range(8)
    )


def test_exact_observation_map_at_zero_retains_the_state_baseline() -> None:
    assert exact_centered_truncated_observation_map(
        ZERO_COORDINATES,
        EXPECTED_DESIGN,
    ) == (F(0),) * 8
    assert exact_truncated_observation_map(
        ZERO_COORDINATES,
        EXPECTED_DESIGN,
    ) == (
        (-1, F(0)),
        (-1, F(0)),
        (-1, F(0)),
        (0, F(0)),
        (0, F(0)),
        (1, F(0)),
        (1, F(0)),
        (1, F(0)),
    )


def test_zero_jacobian_matches_independent_represented_fields() -> None:
    tensor_jacobian = exact_zero_jacobian_from_tensor(EXPECTED_DESIGN)
    field_matrix = exact_joint_state_scale_matrix(EXPECTED_DESIGN)
    assert tensor_jacobian == field_matrix
    assert exact_zero_jacobian(EXPECTED_DESIGN) == field_matrix


def test_hall_and_candidate_pair_ordering_is_deterministic() -> None:
    assert hall_basis_labels() == EXPECTED_LABELS
    assert hall_basis_degrees() == EXPECTED_DEGREES
    assert primary_candidate_state_scale_pairs() == tuple(
        (state_label, F(scale))
        for state_label in ("-log(3)/2", "0", "log(3)/2")
        for scale in (1, 2, 3, 4)
    )


def test_selected_exact_matrix_and_determinant() -> None:
    matrix, determinant, _ = certify_exact_design(EXPECTED_DESIGN)
    assert matrix == EXPECTED_MATRIX
    assert determinant == EXPECTED_DETERMINANT


def test_exact_inverse_is_stored_regression_and_two_sided() -> None:
    matrix, _, inverse = certify_exact_design(EXPECTED_DESIGN)
    assert inverse == EXPECTED_INVERSE
    assert _matrix_product(matrix, inverse) == _identity()
    assert _matrix_product(inverse, matrix) == _identity()


def test_prescribed_search_is_deterministic() -> None:
    assert select_exact_design() == EXPECTED_DESIGN


def test_duplicate_pair_is_rejected() -> None:
    duplicate = EXPECTED_DESIGN[:-1] + (EXPECTED_DESIGN[0],)
    with pytest.raises(ValueError, match="distinct"):
        exact_zero_jacobian(duplicate)
    with pytest.raises(ValueError, match="distinct"):
        exact_truncated_observation_map(ZERO_COORDINATES, duplicate)


def test_removing_any_observation_loses_full_column_rank() -> None:
    for removed in range(8):
        reduced = tuple(
            row for index, row in enumerate(EXPECTED_MATRIX) if index != removed
        )
        assert exact_matrix_rank(reduced) == 7


def test_deliberately_singular_design_is_rejected_exactly() -> None:
    candidates = primary_candidate_state_scale_pairs()
    singular_design = candidates[:8]
    singular_matrix = exact_zero_jacobian(singular_design)
    assert _determinant(singular_matrix) == 0
    assert exact_matrix_rank(singular_matrix) == 7
    with pytest.raises(ValueError, match="exactly singular"):
        certify_exact_design(singular_design)
    with pytest.raises(ValueError, match="only after exact nonsingularity"):
        noncertifying_spectral_diagnostics(singular_matrix)


@pytest.mark.parametrize("multiplier", (F(2), F(-3), F(1, 2)))
def test_common_scale_determinant_has_degree_twenty_two(
    multiplier: Fraction,
) -> None:
    scaled = common_rescale_design(EXPECTED_DESIGN, multiplier)
    scaled_determinant = _determinant(exact_zero_jacobian(scaled))
    assert scaled_determinant == multiplier ** 22 * EXPECTED_DETERMINANT


@pytest.mark.parametrize("scale", (F(-2), F(1), F(3, 2)))
def test_single_nonzero_scale_has_cross_degree_rank_exactly_seven(
    scale: Fraction,
) -> None:
    matrix = single_scale_field_jet_matrix(scale)
    relation = (
        F(0),
        F(0),
        F(4) / scale**2,
        F(3) / scale**3,
        F(0),
        F(0),
        F(0),
        F(2) / scale**4,
    )
    assert tuple(
        sum(
            (entry * coefficient for entry, coefficient in zip(row, relation)),
            start=F(0),
        )
        for row in matrix
    ) == (F(0),) * 7
    assert exact_matrix_rank(matrix) == 7


def test_cross_degree_relation_is_a_represented_field_regression() -> None:
    # Independently derived from D_x=(1-u^2)d/du and
    # [f,g]=(D_x g)f-(D_x f)g, in ascending powers of u=tanh(y).
    h2 = (F(-1, 2), F(0), F(1, 2), F(0), F(0))
    h31 = (F(1, 2), F(0), F(0), F(0), F(-1, 2))
    h43 = (F(1, 4), F(0), F(-1), F(0), F(3, 4))
    polynomial_residual = tuple(
        F(4) * h2[index] + F(3) * h31[index] + F(2) * h43[index]
        for index in range(5)
    )
    assert polynomial_residual == (F(0),) * 5

    matrix = single_scale_field_jet_matrix(F(1))
    first_seven_minor = tuple(tuple(row[:7]) for row in matrix)
    assert _determinant(first_seven_minor) == F(-3, 32)
    assert len({scale for _, scale in EXPECTED_DESIGN}) >= 2


SMALL_LOCAL_COORDINATES = (
    (
        F(1, 1000),
        F(-1, 1200),
        F(1, 1400),
        F(-1, 1600),
        F(1, 1800),
        F(-1, 2000),
        F(1, 2200),
        F(-1, 2400),
    ),
    (
        F(-1, 1500),
        F(1, 1700),
        F(-1, 1900),
        F(1, 2100),
        F(-1, 2300),
        F(1, 2500),
        F(-1, 2700),
        F(1, 2900),
    ),
    (
        F(1, 1300),
        F(1, 1600),
        F(-1, 1900),
        F(-1, 2200),
        F(1, 2500),
        F(1, 2800),
        F(-1, 3100),
        F(-1, 3400),
    ),
)


@pytest.mark.parametrize("flat_coordinates", SMALL_LOCAL_COORDINATES)
def test_independent_truncated_observation_local_numerical_solve(
    flat_coordinates: tuple[Fraction, ...],
) -> None:
    exact_target = _independent_exact_observation(flat_coordinates)
    assert exact_centered_truncated_observation_map(
        _blocks(flat_coordinates),
        EXPECTED_DESIGN,
    ) == exact_target
    numeric_target = np.asarray([float(value) for value in exact_target])
    solved = root(
        lambda candidate: (
            _independent_numeric_observation(candidate) - numeric_target
        ),
        np.zeros(8, dtype=float),
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 1000},
    )
    expected = np.asarray([float(value) for value in flat_coordinates])
    assert solved.success, solved.message
    assert np.allclose(solved.x, expected, rtol=0.0, atol=2e-10)


def test_reproducer_verifies_certificate_without_writing(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research_output" / "reproduce_finite_observation.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "finite_observation_certificate: verified" in completed.stdout
    assert tuple(tmp_path.iterdir()) == before

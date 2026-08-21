from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import lie_reservoir.graded_reconstruction as graded
from lie_reservoir import (
    degree_exact_certificate,
    exact_tanh_witness_graded_jets,
    format_lie_node,
    lyndon_basis,
    reconstruct_tanh_witness_hall_coordinates,
    word_expansion,
)
from lie_reservoir.certificates import (
    _degree_data,
    _differentiate_series,
    _multiply_series,
    _tanh_jet,
)
from research_output import reproduce_graded_reconstruction as reproducer


F = Fraction
ROOT = Path(__file__).resolve().parents[1]

ZERO_COORDINATES = (
    (F(0), F(0)),
    (F(0),),
    (F(0), F(0)),
    (F(0), F(0), F(0)),
)

DENSE_COORDINATES = (
    (F(1), F(-1)),
    (F(2),),
    (F(3), F(-2)),
    (F(-1), F(2), F(-3)),
)

DENSE_JETS = (
    (F(-1, 2), F(0)),
    (F(-5, 4),),
    (F(37, 32), F(3, 2)),
    (F(445, 512), F(5, 32), F(207, 256)),
)

DENSE_CORRECTIONS = (
    (F(0), F(0)),
    (F(-1, 4),),
    (F(1, 8), F(-9, 32)),
    (F(385, 512), F(13, 32), F(81, 256)),
)

DENSE_RESIDUALS = (
    (F(-1, 2), F(0)),
    (F(-1),),
    (F(33, 32), F(57, 32)),
    (F(15, 128), F(-1, 4), F(63, 128)),
)


def _independent_hall_lie_element(
    coordinates: tuple[tuple[Fraction, ...], ...],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for degree, block in enumerate(coordinates, start=1):
        basis = lyndon_basis(input_dim=2, degree=degree)
        for coefficient, node in zip(block, basis, strict=True):
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
    current_order = len(word) - 1
    current = _tanh_jet(
        activation_values[word[-1]],
        slopes[word[-1]],
        current_order,
    )
    for index in reversed(word[:-1]):
        current_order -= 1
        derivative = _differentiate_series(current, current_order)
        field = _tanh_jet(
            activation_values[index],
            slopes[index],
            current_order,
        )
        current = _multiply_series(derivative, field, current_order)
    return current[0]


def _independent_forward(
    coordinates: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    lie_element = _independent_hall_lie_element(coordinates)
    signature = _independent_tensor_exp(lie_element)
    blocks: list[tuple[Fraction, ...]] = []
    for degree in range(1, 5):
        _, _, activation_values, _, _ = _degree_data(degree)
        level = {
            word: coefficient
            for word, coefficient in signature.items()
            if len(word) == degree
        }
        blocks.append(
            tuple(
                sum(
                    (
                        coefficient
                        * _independent_word_value(word, point_values)
                        for word, coefficient in level.items()
                    ),
                    start=F(0),
                )
                for point_values in activation_values
            )
        )
    return tuple(blocks)


def _matrix_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (
                    left[row][index] * right[index][column]
                    for index in range(len(right))
                ),
                start=F(0),
            )
            for column in range(len(right))
        )
        for row in range(len(left))
    )


def _identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(F(int(row == column)) for column in range(size))
        for row in range(size)
    )


def test_tensor_addition_and_truncated_concatenation_product() -> None:
    x1 = {(0,): F(1)}
    x2 = {(1,): F(1)}
    assert graded.tensor_add(x1, {(0,): F(-1), (1,): F(2)}) == {
        (1,): F(2)
    }
    assert graded.truncated_tensor_product(x1, x2) == {(0, 1): F(1)}
    assert graded.truncated_tensor_product(
        {(0, 0, 0, 0): F(1)},
        x2,
    ) == {}


def test_tensor_exponential_through_degree_four_and_factorials() -> None:
    assert graded.tensor_exp({(0,): F(1)}) == {
        (): F(1),
        (0,): F(1),
        (0, 0): F(1, 2),
        (0, 0, 0): F(1, 6),
        (0, 0, 0, 0): F(1, 24),
    }


def test_normalized_scale_jet_has_no_extra_factorial() -> None:
    coordinates = (
        (F(1), F(1)),
        (F(0),),
        (F(0), F(0)),
        (F(0), F(0), F(0)),
    )
    jets = exact_tanh_witness_graded_jets(coordinates)
    # At y=0, W=V1+V2=1/2 and W'=1, so
    # J2=(1/2) D_W^2 Id=(1/2) W'W=1/4.
    assert jets[1] == (F(1, 4),)


def test_word_operator_composition_order() -> None:
    activation_values = (F(0), F(1, 2))
    slopes = (F(1), F(0))
    assert graded._word_operator_value((0, 1), activation_values, slopes) == 0
    assert graded._word_operator_value((1, 0), activation_values, slopes) == F(
        1,
        2,
    )


def test_free_lie_commutator_matches_vector_field_bracket_sign() -> None:
    coordinates = (
        (F(0), F(0)),
        (F(1),),
        (F(0), F(0)),
        (F(0), F(0), F(0)),
    )
    assert exact_tanh_witness_graded_jets(coordinates)[1] == (F(-1, 2),)


def test_lyndon_basis_order_is_the_certified_order() -> None:
    for degree in range(1, 5):
        labels = tuple(
            format_lie_node(node)
            for node in lyndon_basis(input_dim=2, degree=degree)
        )
        assert labels == degree_exact_certificate(degree).basis_labels


def test_recomputed_matrices_match_word_operator_representation() -> None:
    matrices = graded.exact_reconstruction_matrices()
    for degree, matrix in enumerate(matrices, start=1):
        certificate = degree_exact_certificate(degree)
        assert matrix == certificate.exact_minor
        basis = lyndon_basis(input_dim=2, degree=degree)
        _, _, activation_values, _, _ = _degree_data(degree)
        slopes = (F(1), F(0))
        represented = tuple(
            tuple(
                sum(
                    (
                        F(coefficient)
                        * graded._word_operator_value(
                            word,
                            point_values,
                            slopes,
                        )
                        for word, coefficient in word_expansion(node)
                    ),
                    start=F(0),
                )
                for node in basis
            )
            for point_values in activation_values
        )
        assert represented == matrix


def test_exact_inverses_are_two_sided_and_match_regression_guards() -> None:
    expected = (
        ((F(-2), F(2)), (F(2), F(0))),
        ((F(-2),),),
        ((F(16, 15), F(16, 15)), (F(8, 3), F(-8, 3))),
        (
            (F(-64, 45), F(8, 15), F(-64, 45)),
            (F(-16, 3), F(0), F(16, 3)),
            (F(-128, 45), F(76, 15), F(-128, 45)),
        ),
    )
    matrices = graded.exact_reconstruction_matrices()
    inverses = graded.exact_reconstruction_inverses()
    assert inverses == expected
    for matrix, inverse in zip(matrices, inverses, strict=True):
        identity = _identity(len(matrix))
        assert _matrix_product(matrix, inverse) == identity
        assert _matrix_product(inverse, matrix) == identity


def test_every_lower_level_correction_matches_independent_fixture() -> None:
    steps = graded._reconstruction_steps(DENSE_JETS)
    assert tuple(step.lower_level_correction for step in steps) == (
        DENSE_CORRECTIONS
    )
    assert tuple(step.residual for step in steps) == DENSE_RESIDUALS
    assert tuple(step.recovered_hall_coefficients for step in steps) == (
        DENSE_COORDINATES
    )


def test_missing_degree_fails() -> None:
    with pytest.raises(ValueError, match="degrees one through four"):
        exact_tanh_witness_graded_jets(ZERO_COORDINATES[:-1])
    with pytest.raises(ValueError, match="degrees one through four"):
        reconstruct_tanh_witness_hall_coordinates(DENSE_JETS[:-1])


def test_wrong_observation_count_fails() -> None:
    malformed = (DENSE_JETS[0], DENSE_JETS[1], DENSE_JETS[2], (F(1), F(2)))
    with pytest.raises(ValueError, match="degree-4 block must have length 3"):
        reconstruct_tanh_witness_hall_coordinates(malformed)


def test_wrong_basis_order_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original = graded.lyndon_basis

    def reversed_basis(input_dim: int, degree: int):
        return tuple(reversed(original(input_dim=input_dim, degree=degree)))

    monkeypatch.setattr(graded, "lyndon_basis", reversed_basis)
    with pytest.raises(RuntimeError, match="basis ordering is inconsistent"):
        exact_tanh_witness_graded_jets(ZERO_COORDINATES)


def test_deliberately_singular_exact_matrix_fails() -> None:
    singular = ((F(1), F(2)), (F(2), F(4)))
    with pytest.raises(ValueError, match="singular"):
        graded.invert_exact_matrix(singular)


def test_exact_paths_reject_non_fraction_values() -> None:
    float_coordinates = (
        (1.0, F(0)),
        (F(0),),
        (F(0), F(0)),
        (F(0), F(0), F(0)),
    )
    with pytest.raises(TypeError, match="Fraction"):
        exact_tanh_witness_graded_jets(float_coordinates)


def test_unnormalized_scale_derivatives_fail_coordinate_round_trip() -> None:
    coordinates = (
        (F(1), F(1)),
        (F(0),),
        (F(0), F(0)),
        (F(0), F(0), F(0)),
    )
    normalized_jets = exact_tanh_witness_graded_jets(coordinates)
    unnormalized_derivatives = tuple(
        tuple(F(math.factorial(degree)) * value for value in block)
        for degree, block in enumerate(normalized_jets, start=1)
    )
    assert reconstruct_tanh_witness_hall_coordinates(
        unnormalized_derivatives
    ) != coordinates


def test_independent_dense_forward_fixture() -> None:
    assert _independent_forward(DENSE_COORDINATES) == DENSE_JETS
    assert exact_tanh_witness_graded_jets(DENSE_COORDINATES) == DENSE_JETS


def test_decoder_against_independently_fixed_dense_jets() -> None:
    assert reconstruct_tanh_witness_hall_coordinates(DENSE_JETS) == (
        DENSE_COORDINATES
    )


ROUND_TRIP_COORDINATES = (
    ZERO_COORDINATES,
    ((F(2), F(-3)), (F(0),), (F(0), F(0)), (F(0), F(0), F(0))),
    ((F(0), F(0)), (F(-4, 3),), (F(0), F(0)), (F(0), F(0), F(0))),
    ((F(0), F(0)), (F(0),), (F(5, 2), F(-7, 3)), (F(0), F(0), F(0))),
    ((F(0), F(0)), (F(0),), (F(0), F(0)), (F(-2), F(3), F(-5))),
    DENSE_COORDINATES,
    (
        (F(1, 2), F(-2, 3)),
        (F(5, 7),),
        (F(-3, 5), F(4, 9)),
        (F(7, 11), F(-8, 13), F(9, 14)),
    ),
    (
        (F(-5, 3), F(2, 7)),
        (F(-11, 4),),
        (F(13, 8), F(-17, 9)),
        (F(19, 10), F(-23, 12), F(29, 15)),
    ),
)


@pytest.mark.parametrize("coordinates", ROUND_TRIP_COORDINATES)
def test_independent_exact_forward_route_round_trips(
    coordinates: tuple[tuple[Fraction, ...], ...],
) -> None:
    independent_jets = _independent_forward(coordinates)
    assert exact_tanh_witness_graded_jets(coordinates) == independent_jets
    assert reconstruct_tanh_witness_hall_coordinates(
        independent_jets
    ) == coordinates


def test_reproducer_verifies_artifact_without_writing(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research_output" / "reproduce_graded_reconstruction.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "exact_round_trip: verified" in completed.stdout
    assert tuple(tmp_path.iterdir()) == before


def test_reproducer_rejects_a_gate_vector_with_a_zero_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed_gate = (ZERO_COORDINATES,) + reproducer.GATE_COORDINATES[1:]
    monkeypatch.setattr(reproducer, "GATE_COORDINATES", malformed_gate)
    with pytest.raises(RuntimeError, match="all eight coordinates nonzero"):
        reproducer.build_certificate_payload()


def test_source_certificate_hash_is_line_ending_independent(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_content = '{\n  "b": 2,\n  "a": 1\n}\n'
    lf_path.write_bytes(lf_content.encode("utf-8"))
    crlf_path.write_bytes(lf_content.replace("\n", "\r\n").encode("utf-8"))
    assert reproducer._semantic_json_sha256(lf_path) == (
        reproducer._semantic_json_sha256(crlf_path)
    )

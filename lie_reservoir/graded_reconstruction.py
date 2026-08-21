"""Exact graded reconstruction for the certified scalar tanh witness.

The scope is fixed to two generators and homogeneous degrees one through four.
Associative words are chronological tuples of zero-based generator indices:
``(i_1, ..., i_m)`` represents ``X_{i_1 + 1} ... X_{i_m + 1}`` and

``Gamma(i_1 ... i_m) = D_{V_{i_1}} ... D_{V_{i_m}}``.

Thus ``Gamma`` is a homomorphism for concatenation.  With
``D_V f = f' V``, the repository bracket ``[V, W] = W' V - V' W`` is
compatible with ``[X_1, X_2] = X_1 X_2 - X_2 X_1``.  Applying ``Gamma`` to
each homogeneous signature tensor gives the normalized scale jet; no
additional factorial is applied to the response level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

from .certificates import (
    ExactMinor,
    _degree_data,
    _differentiate_series,
    _multiply_series,
    _tanh_jet,
    degree_exact_certificate,
)
from .free_lie import (
    LieNode,
    Word,
    format_lie_node,
    lyndon_basis,
    word_expansion,
)


ExactBlocks: TypeAlias = tuple[tuple[Fraction, ...], ...]
SparseTensor: TypeAlias = dict[Word, Fraction]

_DEGREES = (1, 2, 3, 4)
_INPUT_DIM = 2
_MAX_DEGREE = 4


@dataclass(frozen=True)
class DegreeReconstruction:
    """One exact step of the triangular degree-by-degree reconstruction."""

    degree: int
    basis_labels: tuple[str, ...]
    point_labels: tuple[str, ...]
    observed_jet_vector: tuple[Fraction, ...]
    lower_level_correction: tuple[Fraction, ...]
    residual: tuple[Fraction, ...]
    recovered_hall_coefficients: tuple[Fraction, ...]


def _require_fraction(value: Fraction, *, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must contain only Fraction values.")


def _validate_word(word: Word, *, allow_empty: bool) -> None:
    if not isinstance(word, tuple):
        raise TypeError("tensor words must be tuples.")
    if not allow_empty and not word:
        raise ValueError("the associative word must be nonempty.")
    if len(word) > _MAX_DEGREE:
        raise ValueError("tensor words may have degree at most four.")
    if any(
        isinstance(letter, bool)
        or not isinstance(letter, int)
        or letter < 0
        or letter >= _INPUT_DIM
        for letter in word
    ):
        raise ValueError("tensor word letters must be zero or one.")


def _canonical_tensor(tensor: Mapping[Word, Fraction]) -> SparseTensor:
    if not isinstance(tensor, Mapping):
        raise TypeError("tensor must be a mapping from words to Fractions.")
    result: SparseTensor = {}
    for word, coefficient in tensor.items():
        _validate_word(word, allow_empty=True)
        _require_fraction(coefficient, name="tensor")
        if coefficient != 0:
            result[word] = coefficient
    return dict(sorted(result.items()))


def tensor_add(
    left: Mapping[Word, Fraction],
    right: Mapping[Word, Fraction],
) -> SparseTensor:
    """Add two exact sparse tensors through degree four."""
    result = _canonical_tensor(left)
    for word, coefficient in _canonical_tensor(right).items():
        result[word] = result.get(word, Fraction(0)) + coefficient
        if result[word] == 0:
            del result[word]
    return dict(sorted(result.items()))


def truncated_tensor_product(
    left: Mapping[Word, Fraction],
    right: Mapping[Word, Fraction],
) -> SparseTensor:
    """Return the concatenation product truncated after degree four."""
    left_exact = _canonical_tensor(left)
    right_exact = _canonical_tensor(right)
    result: SparseTensor = {}
    for left_word, left_coefficient in left_exact.items():
        for right_word, right_coefficient in right_exact.items():
            word = left_word + right_word
            if len(word) > _MAX_DEGREE:
                continue
            result[word] = (
                result.get(word, Fraction(0))
                + left_coefficient * right_coefficient
            )
    return {
        word: coefficient
        for word, coefficient in sorted(result.items())
        if coefficient != 0
    }


def _scale_tensor(
    tensor: Mapping[Word, Fraction],
    scalar: Fraction,
) -> SparseTensor:
    _require_fraction(scalar, name="scalar")
    return {
        word: scalar * coefficient
        for word, coefficient in _canonical_tensor(tensor).items()
        if scalar * coefficient != 0
    }


def tensor_exp(lie_element: Mapping[Word, Fraction]) -> SparseTensor:
    """Compute ``exp_tensor(lie_element)`` exactly through degree four."""
    exact_lie_element = _canonical_tensor(lie_element)
    if exact_lie_element.get((), Fraction(0)) != 0:
        raise ValueError("the tensor exponential requires zero scalar part.")

    unit = {(): Fraction(1)}
    result = unit
    power = unit
    factorial = 1
    for exponent in range(1, _MAX_DEGREE + 1):
        power = truncated_tensor_product(power, exact_lie_element)
        factorial *= exponent
        result = tensor_add(
            result,
            _scale_tensor(power, Fraction(1, factorial)),
        )
    return result


def _degree_context(
    degree: int,
) -> tuple[
    tuple[LieNode, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[Fraction, Fraction], ...],
    tuple[Fraction, Fraction],
    ExactMinor,
]:
    if degree not in _DEGREES:
        raise ValueError("degree must be one of 1, 2, 3, or 4.")

    certificate = degree_exact_certificate(degree)
    (
        data_basis_labels,
        point_labels,
        activation_values,
        _,
        _,
    ) = _degree_data(degree)
    basis = lyndon_basis(input_dim=_INPUT_DIM, degree=degree)
    basis_labels = tuple(format_lie_node(node) for node in basis)

    if basis_labels != certificate.basis_labels:
        raise RuntimeError(
            f"the degree-{degree} Lyndon basis ordering is inconsistent."
        )
    if data_basis_labels != certificate.basis_labels:
        raise RuntimeError(
            f"the degree-{degree} certificate basis labels are inconsistent."
        )
    if point_labels != certificate.evaluation_points:
        raise RuntimeError(
            f"the degree-{degree} certificate point order is inconsistent."
        )

    try:
        slopes = tuple(Fraction(entry) for entry in certificate.weight_entries)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError("certificate weights are not exact rationals.") from error
    if len(slopes) != _INPUT_DIM:
        raise RuntimeError("certificate field count is inconsistent.")
    exact_slopes = (slopes[0], slopes[1])

    matrix = certificate.exact_minor
    if len(matrix) != len(point_labels) or any(
        len(row) != len(basis) for row in matrix
    ):
        raise RuntimeError(
            f"the degree-{degree} certificate matrix shape is inconsistent."
        )
    if len(matrix) != len(basis):
        raise RuntimeError(
            f"the degree-{degree} certificate matrix is not square."
        )
    return (
        basis,
        basis_labels,
        point_labels,
        activation_values,
        exact_slopes,
        matrix,
    )


def _expected_block_sizes() -> tuple[int, ...]:
    return tuple(len(_degree_context(degree)[0]) for degree in _DEGREES)


def _validate_blocks(blocks: ExactBlocks, *, name: str) -> ExactBlocks:
    if not isinstance(blocks, tuple):
        raise TypeError(f"{name} must be a tuple of four tuples.")
    if len(blocks) != len(_DEGREES):
        raise ValueError(f"{name} must contain degrees one through four.")

    expected_sizes = _expected_block_sizes()
    validated: list[tuple[Fraction, ...]] = []
    for degree, (block, expected_size) in enumerate(
        zip(blocks, expected_sizes, strict=True),
        start=1,
    ):
        if not isinstance(block, tuple):
            raise TypeError(f"{name} degree-{degree} block must be a tuple.")
        if len(block) != expected_size:
            raise ValueError(
                f"{name} degree-{degree} block must have length "
                f"{expected_size}."
            )
        for value in block:
            _require_fraction(value, name=name)
        validated.append(block)
    return tuple(validated)


def _hall_lie_element_from_validated(coordinates: ExactBlocks) -> SparseTensor:
    result: SparseTensor = {}
    for degree, coordinate_block in zip(_DEGREES, coordinates, strict=True):
        basis = _degree_context(degree)[0]
        for coefficient, node in zip(coordinate_block, basis, strict=True):
            for word, integer_coefficient in word_expansion(node):
                value = coefficient * integer_coefficient
                result[word] = result.get(word, Fraction(0)) + value
    return {
        word: coefficient
        for word, coefficient in sorted(result.items())
        if coefficient != 0
    }


def _word_operator_jet(
    word: Word,
    activation_values: tuple[Fraction, Fraction],
    slopes: tuple[Fraction, Fraction],
    order: int,
) -> tuple[Fraction, ...]:
    """Evaluate ``Gamma(word) Id`` as a normalized local Taylor jet."""
    if len(word) == 1:
        index = word[0]
        return _tanh_jet(activation_values[index], slopes[index], order)

    inner = _word_operator_jet(
        word[1:],
        activation_values,
        slopes,
        order + 1,
    )
    derivative = _differentiate_series(inner, order)
    outer_field = _tanh_jet(
        activation_values[word[0]],
        slopes[word[0]],
        order,
    )
    return _multiply_series(derivative, outer_field, order)


def _word_operator_value(
    word: Word,
    activation_values: tuple[Fraction, Fraction],
    slopes: tuple[Fraction, Fraction],
) -> Fraction:
    _validate_word(word, allow_empty=False)
    return _word_operator_jet(word, activation_values, slopes, order=0)[0]


def _homogeneous_level(
    tensor: Mapping[Word, Fraction],
    degree: int,
) -> SparseTensor:
    if degree not in _DEGREES:
        raise ValueError("degree must be one of 1, 2, 3, or 4.")
    return {
        word: coefficient
        for word, coefficient in _canonical_tensor(tensor).items()
        if len(word) == degree
    }


def _evaluate_signature_level(
    degree: int,
    signature_level: Mapping[Word, Fraction],
) -> tuple[Fraction, ...]:
    exact_level = _canonical_tensor(signature_level)
    if any(len(word) != degree for word in exact_level):
        raise ValueError("signature_level must be homogeneous of the given degree.")

    _, _, _, activation_values, slopes, _ = _degree_context(degree)
    return tuple(
        sum(
            (
                coefficient
                * _word_operator_value(word, point_values, slopes)
                for word, coefficient in exact_level.items()
            ),
            start=Fraction(0),
        )
        for point_values in activation_values
    )


def exact_tanh_witness_graded_jets(
    coordinates: ExactBlocks,
) -> ExactBlocks:
    """Encode Hall log-signature coordinates as exact normalized scale jets."""
    exact_coordinates = _validate_blocks(coordinates, name="coordinates")
    lie_element = _hall_lie_element_from_validated(exact_coordinates)
    signature = tensor_exp(lie_element)
    return tuple(
        _evaluate_signature_level(
            degree,
            _homogeneous_level(signature, degree),
        )
        for degree in _DEGREES
    )


def exact_reconstruction_matrices() -> tuple[ExactMinor, ...]:
    """Recompute the four matrices with the authoritative certificate path."""
    return tuple(_degree_context(degree)[5] for degree in _DEGREES)


def invert_exact_matrix(matrix: ExactMinor) -> ExactMinor:
    """Invert one nonempty square matrix of Fractions by Gauss--Jordan."""
    if not isinstance(matrix, tuple) or not matrix:
        raise ValueError("matrix must be a nonempty tuple of rows.")
    size = len(matrix)
    if size > 3:
        raise ValueError("only the certified matrix sizes one through three apply.")
    if any(not isinstance(row, tuple) or len(row) != size for row in matrix):
        raise ValueError("matrix must be square with tuple rows.")
    for row in matrix:
        for value in row:
            _require_fraction(value, name="matrix")

    work = [
        list(row)
        + [Fraction(int(row_index == column)) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot_row = next(
            (
                row
                for row in range(column, size)
                if work[row][column] != 0
            ),
            None,
        )
        if pivot_row is None:
            raise ValueError("matrix is singular over the rationals.")
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]

        pivot = work[column][column]
        work[column] = [value / pivot for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor == 0:
                continue
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    work[row],
                    work[column],
                    strict=True,
                )
            ]
    return tuple(
        tuple(row[size:])
        for row in work
    )


def exact_reconstruction_inverses() -> tuple[ExactMinor, ...]:
    """Recompute the four exact inverses from the authoritative matrices."""
    return tuple(
        invert_exact_matrix(matrix)
        for matrix in exact_reconstruction_matrices()
    )


def _matrix_vector_product(
    matrix: ExactMinor,
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix and vector dimensions are inconsistent.")
    return tuple(
        sum(
            (entry * value for entry, value in zip(row, vector, strict=True)),
            start=Fraction(0),
        )
        for row in matrix
    )


def _zero_coordinate_blocks() -> ExactBlocks:
    return tuple(
        tuple(Fraction(0) for _ in _degree_context(degree)[0])
        for degree in _DEGREES
    )


def _lower_level_correction(
    degree: int,
    recovered_blocks: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    if len(recovered_blocks) != degree - 1:
        raise ValueError("recovered lower-degree blocks are incomplete.")
    zero_blocks = _zero_coordinate_blocks()
    lower_only = tuple(
        recovered_blocks[index]
        if index < len(recovered_blocks)
        else zero_blocks[index]
        for index in range(len(_DEGREES))
    )
    lower_lie_element = _hall_lie_element_from_validated(lower_only)
    lower_signature = tensor_exp(lower_lie_element)
    return _evaluate_signature_level(
        degree,
        _homogeneous_level(lower_signature, degree),
    )


def _reconstruction_steps(jets: ExactBlocks) -> tuple[DegreeReconstruction, ...]:
    exact_jets = _validate_blocks(jets, name="jets")
    matrices = exact_reconstruction_matrices()
    recovered: list[tuple[Fraction, ...]] = []
    steps: list[DegreeReconstruction] = []

    for degree, observed, matrix in zip(
        _DEGREES,
        exact_jets,
        matrices,
        strict=True,
    ):
        correction = _lower_level_correction(degree, tuple(recovered))
        residual = tuple(
            observed_value - correction_value
            for observed_value, correction_value in zip(
                observed,
                correction,
                strict=True,
            )
        )
        inverse = invert_exact_matrix(matrix)
        coefficients = _matrix_vector_product(inverse, residual)
        _, basis_labels, point_labels, _, _, _ = _degree_context(degree)
        steps.append(
            DegreeReconstruction(
                degree=degree,
                basis_labels=basis_labels,
                point_labels=point_labels,
                observed_jet_vector=observed,
                lower_level_correction=correction,
                residual=residual,
                recovered_hall_coefficients=coefficients,
            )
        )
        recovered.append(coefficients)
    return tuple(steps)


def reconstruct_tanh_witness_hall_coordinates(
    jets: ExactBlocks,
) -> ExactBlocks:
    """Recover all eight Hall coordinates recursively from ideal graded jets."""
    return tuple(
        step.recovered_hall_coefficients
        for step in _reconstruction_steps(jets)
    )


__all__ = [
    "DegreeReconstruction",
    "ExactBlocks",
    "SparseTensor",
    "exact_reconstruction_inverses",
    "exact_reconstruction_matrices",
    "exact_tanh_witness_graded_jets",
    "invert_exact_matrix",
    "reconstruct_tanh_witness_hall_coordinates",
    "tensor_add",
    "tensor_exp",
    "truncated_tensor_product",
]

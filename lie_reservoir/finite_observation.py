"""Exact finite observations of the certified degree-four truncated response.

This module concerns only object B,

``F_y^(4)(r; z) = y + sum_{m=1}^4 r^m Gamma(S_m(z)) Id(y)``.

At the certified states ``y = +/-q``, the baseline is not rational.  The
centered response is Fraction-valued, while an uncentered exact value is
represented by ``(q_multiplier, rational_increment)`` and denotes
``q_multiplier * q + rational_increment`` for ``q = log(3) / 2``.
No full CDE response or level-five remainder is represented here.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import TypeAlias

from .certificates import (
    ExactMinor,
    _determinant,
    _represented_jet,
)
from .free_lie import LieNode, Word, word_expansion
from .graded_reconstruction import (
    ExactBlocks,
    SparseTensor,
    _degree_context,
    _require_fraction,
    _signature_from_hall_coordinates,
    _word_operator_value,
    invert_exact_matrix,
)


ObservationPair: TypeAlias = tuple[str, Fraction]
ObservationDesign: TypeAlias = tuple[ObservationPair, ...]
ExactTruncatedValue: TypeAlias = tuple[int, Fraction]
ExactObservationVector: TypeAlias = tuple[ExactTruncatedValue, ...]
ExactMatrix: TypeAlias = tuple[tuple[Fraction, ...], ...]

_DEGREES = (1, 2, 3, 4)
_PRIMARY_SCALES = (Fraction(1), Fraction(2), Fraction(3), Fraction(4))
_FALLBACK_SCALES = (
    Fraction(-3),
    Fraction(-2),
    Fraction(-1),
    Fraction(1),
    Fraction(2),
    Fraction(3),
)


def _state_context() -> tuple[
    tuple[str, tuple[Fraction, Fraction], int],
    ...,
]:
    _, _, state_labels, activation_values, _, _ = _degree_context(4)
    if len(state_labels) != 3:
        raise RuntimeError("the certified state set is not three-dimensional.")

    q_multipliers: list[int] = []
    for values in activation_values:
        multiplier = 2 * values[0]
        if multiplier.denominator != 1 or multiplier.numerator not in {-1, 0, 1}:
            raise RuntimeError(
                "the certified states are inconsistent with q = log(3) / 2."
            )
        q_multipliers.append(multiplier.numerator)
    return tuple(
        (label, values, multiplier)
        for label, values, multiplier in zip(
            state_labels,
            activation_values,
            q_multipliers,
            strict=True,
        )
    )


def _hall_context() -> tuple[
    tuple[LieNode, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    nodes: list[LieNode] = []
    labels: list[str] = []
    degrees: list[int] = []
    for degree in _DEGREES:
        degree_nodes, degree_labels, _, _, _, _ = _degree_context(degree)
        nodes.extend(degree_nodes)
        labels.extend(degree_labels)
        degrees.extend(degree for _ in degree_nodes)
    return tuple(nodes), tuple(labels), tuple(degrees)


def hall_basis_labels() -> tuple[str, ...]:
    """Return the deterministic degree-then-Lyndon column ordering."""
    return _hall_context()[1]


def hall_basis_degrees() -> tuple[int, ...]:
    """Return the degree attached to every ordered Hall column."""
    return _hall_context()[2]


def _state_lookup() -> dict[str, tuple[tuple[Fraction, Fraction], int]]:
    return {
        label: (activation_values, q_multiplier)
        for label, activation_values, q_multiplier in _state_context()
    }


def _validate_pairs(
    pairs: ObservationDesign,
    *,
    required_size: int | None,
) -> ObservationDesign:
    if not isinstance(pairs, tuple):
        raise TypeError("state-scale pairs must be a tuple.")
    if required_size is not None and len(pairs) != required_size:
        raise ValueError(f"the design must contain exactly {required_size} pairs.")
    if not pairs:
        raise ValueError("at least one state-scale pair is required.")

    allowed_states = _state_lookup()
    validated: list[ObservationPair] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("each state-scale pair must be a two-tuple.")
        state_label, scale = pair
        if not isinstance(state_label, str):
            raise TypeError("state labels must be strings.")
        if state_label not in allowed_states:
            raise ValueError("state label is not in the certified three-point set.")
        _require_fraction(scale, name="scales")
        if scale == 0:
            raise ValueError("observation scales must be nonzero.")
        validated.append((state_label, scale))
    if len(set(validated)) != len(validated):
        raise ValueError("state-scale pairs must be distinct.")
    return tuple(validated)


def candidate_state_scale_pairs(
    scales: tuple[Fraction, ...],
) -> ObservationDesign:
    """Return the state-major candidate order for one exact scale tuple."""
    if not isinstance(scales, tuple) or not scales:
        raise TypeError("scales must be a nonempty tuple of Fractions.")
    for scale in scales:
        _require_fraction(scale, name="scales")
        if scale == 0:
            raise ValueError("candidate scales must be nonzero.")
    if len(set(scales)) != len(scales):
        raise ValueError("candidate scales must be distinct.")
    pairs = tuple(
        (state_label, scale)
        for state_label, _, _ in _state_context()
        for scale in scales
    )
    return _validate_pairs(pairs, required_size=None)


def primary_candidate_state_scale_pairs() -> ObservationDesign:
    """Return the prescribed twelve-row positive-scale candidate set."""
    return candidate_state_scale_pairs(_PRIMARY_SCALES)


def fallback_candidate_state_scale_pairs() -> ObservationDesign:
    """Return the prescribed signed fallback candidate set."""
    return candidate_state_scale_pairs(_FALLBACK_SCALES)


def _centered_response_from_signature(
    signature: SparseTensor,
    pair: ObservationPair,
) -> Fraction:
    state_label, scale = pair
    activation_values = _state_lookup()[state_label][0]
    slopes = _degree_context(4)[4]
    return sum(
        (
            coefficient
            * scale ** len(word)
            * _word_operator_value(word, activation_values, slopes)
            for word, coefficient in signature.items()
            if word
        ),
        start=Fraction(0),
    )


def exact_centered_truncated_observation_map(
    coordinates: ExactBlocks,
    design: ObservationDesign,
) -> tuple[Fraction, ...]:
    """Evaluate ``Phi_D(z) - Phi_D(0)`` exactly over the rationals."""
    exact_design = _validate_pairs(design, required_size=8)
    signature = _signature_from_hall_coordinates(coordinates)
    return tuple(
        _centered_response_from_signature(signature, pair)
        for pair in exact_design
    )


def exact_truncated_observation_map(
    coordinates: ExactBlocks,
    design: ObservationDesign,
) -> ExactObservationVector:
    """Evaluate the literal truncated map with an exact ``q`` representation."""
    exact_design = _validate_pairs(design, required_size=8)
    increments = exact_centered_truncated_observation_map(
        coordinates,
        exact_design,
    )
    state_data = _state_lookup()
    return tuple(
        (state_data[state_label][1], increment)
        for (state_label, _), increment in zip(
            exact_design,
            increments,
            strict=True,
        )
    )


def _tensor_linearized_rows(pairs: ObservationDesign) -> ExactMatrix:
    nodes, _, degrees = _hall_context()
    slopes = _degree_context(4)[4]
    state_data = _state_lookup()
    expansions: tuple[tuple[tuple[Word, int], ...], ...] = tuple(
        word_expansion(node) for node in nodes
    )
    return tuple(
        tuple(
            scale ** degree
            * sum(
                (
                    Fraction(integer_coefficient)
                    * _word_operator_value(
                        word,
                        state_data[state_label][0],
                        slopes,
                    )
                    for word, integer_coefficient in expansion
                ),
                start=Fraction(0),
            )
            for degree, expansion in zip(degrees, expansions, strict=True)
        )
        for state_label, scale in pairs
    )


def _represented_field_rows(pairs: ObservationDesign) -> ExactMatrix:
    nodes, _, degrees = _hall_context()
    slopes = _degree_context(4)[4]
    state_data = _state_lookup()
    return tuple(
        tuple(
            scale ** degree
            * _represented_jet(
                node,
                state_data[state_label][0],
                slopes,
                order=0,
            )[0]
            for node, degree in zip(nodes, degrees, strict=True)
        )
        for state_label, scale in pairs
    )


def _verified_linearized_rows(pairs: ObservationDesign) -> ExactMatrix:
    tensor_rows = _tensor_linearized_rows(pairs)
    represented_rows = _represented_field_rows(pairs)
    if tensor_rows != represented_rows:
        raise RuntimeError(
            "the tensor-map and represented-field Jacobians disagree."
        )
    return tensor_rows


def exact_zero_jacobian_from_tensor(
    design: ObservationDesign,
) -> ExactMinor:
    """Differentiate the tensor-polynomial observation map at zero."""
    exact_design = _validate_pairs(design, required_size=8)
    return _tensor_linearized_rows(exact_design)


def exact_joint_state_scale_matrix(
    design: ObservationDesign,
) -> ExactMinor:
    """Compute ``r^degree rho(h)(y)`` by recursive Lie representation."""
    exact_design = _validate_pairs(design, required_size=8)
    return _represented_field_rows(exact_design)


def exact_zero_jacobian(design: ObservationDesign) -> ExactMinor:
    """Return the zero Jacobian after requiring both exact constructions."""
    exact_design = _validate_pairs(design, required_size=8)
    return _verified_linearized_rows(exact_design)


def _matrix_shape(matrix: ExactMatrix) -> tuple[int, int]:
    if not isinstance(matrix, tuple) or not matrix:
        raise ValueError("matrix must be a nonempty tuple of tuple rows.")
    if any(not isinstance(row, tuple) for row in matrix):
        raise TypeError("matrix rows must be tuples.")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have one common nonzero length.")
    for row in matrix:
        for value in row:
            _require_fraction(value, name="matrix")
    return len(matrix), width


def exact_matrix_rank(matrix: ExactMatrix) -> int:
    """Return the exact rank of one small rectangular Fraction matrix."""
    row_count, column_count = _matrix_shape(matrix)
    work = [list(row) for row in matrix]
    rank = 0
    for column in range(column_count):
        pivot_row = next(
            (
                row
                for row in range(rank, row_count)
                if work[row][column] != 0
            ),
            None,
        )
        if pivot_row is None:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        pivot = work[rank][column]
        work[rank] = [value / pivot for value in work[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            factor = work[row][column]
            if factor == 0:
                continue
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    work[row],
                    work[rank],
                    strict=True,
                )
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def _matrix_product(left: ExactMinor, right: ExactMinor) -> ExactMinor:
    if len(left) != len(right):
        raise ValueError("square matrix dimensions do not agree.")
    size = len(left)
    return tuple(
        tuple(
            sum(
                (
                    left[row][index] * right[index][column]
                    for index in range(size)
                ),
                start=Fraction(0),
            )
            for column in range(size)
        )
        for row in range(size)
    )


def _identity(size: int) -> ExactMinor:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(size))
        for row in range(size)
    )


def certify_exact_design(
    design: ObservationDesign,
) -> tuple[ExactMinor, Fraction, ExactMinor]:
    """Return exact matrix, nonzero determinant, and verified inverse."""
    matrix = exact_zero_jacobian(design)
    determinant = _determinant(matrix)
    if determinant == 0:
        raise ValueError("the state-scale design is exactly singular.")
    inverse = invert_exact_matrix(matrix)
    identity = _identity(len(matrix))
    if (
        _matrix_product(matrix, inverse) != identity
        or _matrix_product(inverse, matrix) != identity
    ):
        raise RuntimeError("the exact inverse failed a two-sided identity check.")
    return matrix, determinant, inverse


def noncertifying_spectral_diagnostics(
    matrix: ExactMinor,
) -> tuple[tuple[float, ...], float]:
    """Return binary64 singular values and condition number as diagnostics."""
    row_count, column_count = _matrix_shape(matrix)
    if row_count != column_count:
        raise ValueError("spectral diagnostics require a square matrix.")
    if _determinant(matrix) == 0:
        raise ValueError(
            "spectral diagnostics apply only after exact nonsingularity."
        )
    import numpy as np

    numeric = np.asarray(
        [[float(value) for value in row] for row in matrix],
        dtype=np.float64,
    )
    singular_values = np.linalg.svd(numeric, compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1])
    return tuple(float(value) for value in singular_values), condition_number


def _scale_complexity(
    design: ObservationDesign,
) -> tuple[int, tuple[Fraction, ...]]:
    distinct_magnitudes = tuple(
        sorted({abs(scale) for _, scale in design})
    )
    return len(distinct_magnitudes), distinct_magnitudes


def _pair_order_key(
    design: ObservationDesign,
) -> tuple[tuple[int, Fraction], ...]:
    state_order = {
        label: index
        for index, (label, _, _) in enumerate(_state_context())
    }
    return tuple(
        (state_order[state_label], scale)
        for state_label, scale in design
    )


def _search_grid(
    scales: tuple[Fraction, ...],
) -> tuple[ObservationDesign | None, int, int]:
    candidates = candidate_state_scale_pairs(scales)
    candidate_rows = _verified_linearized_rows(candidates)
    best_design: ObservationDesign | None = None
    best_key: tuple[object, ...] | None = None
    subset_count = 0
    certified_count = 0

    for indices in combinations(range(len(candidates)), 8):
        subset_count += 1
        matrix = tuple(candidate_rows[index] for index in indices)
        determinant = _determinant(matrix)
        if determinant == 0:
            continue
        certified_count += 1
        design = tuple(candidates[index] for index in indices)
        _, condition_number = noncertifying_spectral_diagnostics(matrix)
        selection_key = (
            _scale_complexity(design),
            condition_number,
            _pair_order_key(design),
        )
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_design = design
    return best_design, subset_count, certified_count


def _prescribed_design_search(
) -> tuple[ObservationDesign, str, int, int]:
    primary, subset_count, certified_count = _search_grid(_PRIMARY_SCALES)
    if primary is not None:
        return primary, "primary", subset_count, certified_count

    fallback, subset_count, certified_count = _search_grid(_FALLBACK_SCALES)
    if fallback is not None:
        return fallback, "fallback", subset_count, certified_count
    raise RuntimeError("neither prescribed exact search grid has full rank.")


def select_exact_design() -> ObservationDesign:
    """Run the prescribed deterministic exact search and return its design."""
    return _prescribed_design_search()[0]


def common_rescale_design(
    design: ObservationDesign,
    multiplier: Fraction,
) -> ObservationDesign:
    """Multiply every scale in an exact design by one common nonzero value."""
    exact_design = _validate_pairs(design, required_size=8)
    _require_fraction(multiplier, name="multiplier")
    if multiplier == 0:
        raise ValueError("the common scale multiplier must be nonzero.")
    return tuple(
        (state_label, multiplier * scale)
        for state_label, scale in exact_design
    )


def single_scale_field_jet_matrix(
    scale: Fraction,
) -> ExactMatrix:
    """Return orders zero through six at state zero for all eight fields."""
    _require_fraction(scale, name="scale")
    if scale == 0:
        raise ValueError("the common scale must be nonzero.")
    nodes, _, degrees = _hall_context()
    state_data = _state_lookup()
    zero_label = next(
        label
        for label, _, q_multiplier in _state_context()
        if q_multiplier == 0
    )
    activation_values = state_data[zero_label][0]
    slopes = _degree_context(4)[4]
    columns = tuple(
        tuple(
            scale ** degree * coefficient
            for coefficient in _represented_jet(
                node,
                activation_values,
                slopes,
                order=6,
            )
        )
        for node, degree in zip(nodes, degrees, strict=True)
    )
    return tuple(
        tuple(columns[column][order] for column in range(len(columns)))
        for order in range(7)
    )


__all__ = [
    "ExactMatrix",
    "ExactObservationVector",
    "ExactTruncatedValue",
    "ObservationDesign",
    "ObservationPair",
    "candidate_state_scale_pairs",
    "certify_exact_design",
    "common_rescale_design",
    "exact_centered_truncated_observation_map",
    "exact_joint_state_scale_matrix",
    "exact_matrix_rank",
    "exact_truncated_observation_map",
    "exact_zero_jacobian",
    "exact_zero_jacobian_from_tensor",
    "fallback_candidate_state_scale_pairs",
    "hall_basis_degrees",
    "hall_basis_labels",
    "noncertifying_spectral_diagnostics",
    "primary_candidate_state_scale_pairs",
    "select_exact_design",
    "single_scale_field_jet_matrix",
]

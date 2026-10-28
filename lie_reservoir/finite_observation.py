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
from itertools import combinations, product
from math import factorial, prod
from typing import TypeAlias

from .certificates import (
    ExactMinor,
    _determinant,
    _represented_jet,
)
from .free_lie import LieNode, Word, word_expansion
from .graded_reconstruction import (
    ExactBlocks,
    _degree_context,
    _require_fraction,
    _validate_blocks,
    _word_operator_value,
    invert_exact_matrix,
    truncated_tensor_product,
)


ObservationPair: TypeAlias = tuple[str, Fraction]
ObservationDesign: TypeAlias = tuple[ObservationPair, ...]
ExactTruncatedValue: TypeAlias = tuple[int, Fraction]
ExactObservationVector: TypeAlias = tuple[ExactTruncatedValue, ...]
ExactMatrix: TypeAlias = tuple[tuple[Fraction, ...], ...]
HallMultiIndex: TypeAlias = tuple[int, ...]
ExactHallPolynomial: TypeAlias = dict[HallMultiIndex, Fraction]
ExactHallPolynomialMatrix: TypeAlias = tuple[
    tuple[ExactHallPolynomial, ...], ...
]
ExactRadiusEntry: TypeAlias = tuple[Fraction, Fraction, Fraction, bool]

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
_STABILITY_RADII = tuple(Fraction(1, 2**power) for power in range(1, 17))


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


def _hall_tensor_expansions() -> tuple[dict[Word, Fraction], ...]:
    expansions: list[dict[Word, Fraction]] = []
    for node in _hall_context()[0]:
        expansion: dict[Word, Fraction] = {}
        for word, integer_coefficient in word_expansion(node):
            expansion[word] = (
                expansion.get(word, Fraction(0))
                + Fraction(integer_coefficient)
            )
        expansions.append(
            {
                word: coefficient
                for word, coefficient in sorted(expansion.items())
                if coefficient != 0
            }
        )
    return tuple(expansions)


def _hall_multiindex(sequence: tuple[int, ...]) -> HallMultiIndex:
    return tuple(sequence.count(index) for index in range(8))


def exact_centered_observation_polynomials(
    design: ObservationDesign,
) -> tuple[ExactHallPolynomial, ...]:
    """Return the eight exact Hall-coordinate response polynomials.

    The scalar baseline is omitted.  A dictionary key is an eight-entry
    multi-index in deterministic Hall order.  The construction expands the
    existing Lyndon word tensors inside ``exp_tensor(ell)`` and is restricted
    to the certified truncation degree four.
    """
    exact_design = _validate_pairs(design, required_size=8)
    _, _, degrees = _hall_context()
    hall_tensors = _hall_tensor_expansions()
    state_data = _state_lookup()
    slopes = _degree_context(4)[4]
    polynomials: list[ExactHallPolynomial] = [dict() for _ in exact_design]

    for exponent in range(1, 5):
        divisor = Fraction(factorial(exponent))
        for sequence in product(range(8), repeat=exponent):
            tensor_degree = sum(degrees[index] for index in sequence)
            if tensor_degree > 4:
                continue
            tensor: dict[Word, Fraction] = {(): Fraction(1)}
            for index in sequence:
                tensor = truncated_tensor_product(
                    tensor,
                    hall_tensors[index],
                )
            multiindex = _hall_multiindex(sequence)
            for row, (state_label, scale) in enumerate(exact_design):
                activation_values = state_data[state_label][0]
                response_coefficient = sum(
                    (
                        word_coefficient
                        * scale ** len(word)
                        * _word_operator_value(
                            word,
                            activation_values,
                            slopes,
                        )
                        for word, word_coefficient in tensor.items()
                        if word
                    ),
                    start=Fraction(0),
                ) / divisor
                if response_coefficient != 0:
                    polynomials[row][multiindex] = (
                        polynomials[row].get(multiindex, Fraction(0))
                        + response_coefficient
                    )

    return tuple(
        {
            multiindex: coefficient
            for multiindex, coefficient in sorted(polynomial.items())
            if coefficient != 0
        }
        for polynomial in polynomials
    )


def _flatten_hall_coordinates(coordinates: ExactBlocks) -> tuple[Fraction, ...]:
    exact_coordinates = _validate_blocks(coordinates, name="coordinates")
    return tuple(
        coefficient
        for block in exact_coordinates
        for coefficient in block
    )


def _evaluate_hall_polynomial(
    polynomial: ExactHallPolynomial,
    coordinates: tuple[Fraction, ...],
) -> Fraction:
    if len(coordinates) != 8:
        raise ValueError("Hall coordinate vectors must have length eight.")
    return sum(
        (
            coefficient
            * prod(
                coordinate**exponent
                for coordinate, exponent in zip(
                    coordinates,
                    multiindex,
                    strict=True,
                )
            )
            for multiindex, coefficient in polynomial.items()
        ),
        start=Fraction(0),
    )


def exact_centered_truncated_observation_map(
    coordinates: ExactBlocks,
    design: ObservationDesign,
) -> tuple[Fraction, ...]:
    """Evaluate ``Phi_D(z) - Phi_D(0)`` exactly over the rationals."""
    exact_design = _validate_pairs(design, required_size=8)
    flat_coordinates = _flatten_hall_coordinates(coordinates)
    return tuple(
        _evaluate_hall_polynomial(polynomial, flat_coordinates)
        for polynomial in exact_centered_observation_polynomials(exact_design)
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


def exact_jacobian_difference_polynomials(
    design: ObservationDesign,
) -> ExactHallPolynomialMatrix:
    """Return the exact entries of ``D Phi(z) - D Phi(0)``.

    Each entry uses eight-entry Hall multi-indices.  The zero constant term is
    checked against the independently certified zero Jacobian and then removed.
    """
    exact_design = _validate_pairs(design, required_size=8)
    response_polynomials = exact_centered_observation_polynomials(exact_design)
    zero_multiindex = (0,) * 8
    jacobian: list[list[ExactHallPolynomial]] = [
        [dict() for _ in range(8)] for _ in exact_design
    ]

    for row, response_polynomial in enumerate(response_polynomials):
        for multiindex, coefficient in response_polynomial.items():
            for column, exponent in enumerate(multiindex):
                if exponent == 0:
                    continue
                derivative_index = list(multiindex)
                derivative_index[column] -= 1
                key = tuple(derivative_index)
                jacobian[row][column][key] = (
                    jacobian[row][column].get(key, Fraction(0))
                    + coefficient * exponent
                )

    certified = exact_zero_jacobian(exact_design)
    result: list[tuple[ExactHallPolynomial, ...]] = []
    for row in range(8):
        result_row: list[ExactHallPolynomial] = []
        for column in range(8):
            polynomial = jacobian[row][column]
            constant = polynomial.pop(zero_multiindex, Fraction(0))
            if constant != certified[row][column]:
                raise RuntimeError(
                    "the response-polynomial derivative disagrees with the "
                    "certified zero Jacobian."
                )
            result_row.append(
                {
                    multiindex: value
                    for multiindex, value in sorted(polynomial.items())
                    if value != 0
                }
            )
        result.append(tuple(result_row))
    return tuple(result)


def exact_jacobian_row_degree_sums(
    design: ObservationDesign,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    """Group absolute Jacobian-variation coefficients by degree one to three."""
    polynomial_matrix = exact_jacobian_difference_polynomials(design)
    rows: list[tuple[Fraction, Fraction, Fraction]] = []
    for matrix_row in polynomial_matrix:
        degree_sums = [Fraction(0), Fraction(0), Fraction(0)]
        for polynomial in matrix_row:
            for multiindex, coefficient in polynomial.items():
                degree = sum(multiindex)
                if degree not in {1, 2, 3}:
                    raise RuntimeError(
                        "the degree-four response has an invalid Jacobian degree."
                    )
                degree_sums[degree - 1] += abs(coefficient)
        rows.append(tuple(degree_sums))
    return tuple(rows)


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


def exact_matrix_infinity_norm(matrix: ExactMatrix) -> Fraction:
    """Return the induced infinity norm of an exact matrix."""
    _matrix_shape(matrix)
    return max(
        sum((abs(value) for value in row), start=Fraction(0))
        for row in matrix
    )


def exact_inverse_row_sums(inverse: ExactMinor) -> tuple[Fraction, ...]:
    """Return the absolute row sums of one exact inverse matrix."""
    row_count, column_count = _matrix_shape(inverse)
    if row_count != column_count:
        raise ValueError("inverse row sums require a square matrix.")
    return tuple(
        sum((abs(value) for value in row), start=Fraction(0))
        for row in inverse
    )


def stability_test_radii() -> tuple[Fraction, ...]:
    """Return the complete prescribed decreasing dyadic radius list."""
    return _STABILITY_RADII


def exact_jacobian_variation_bound(
    design: ObservationDesign,
    radius: Fraction,
) -> Fraction:
    """Return the exact coefficient-sum majorant ``Delta(radius)``."""
    _require_fraction(radius, name="radius")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    row_degree_sums = exact_jacobian_row_degree_sums(design)
    return _variation_bound_from_row_degree_sums(row_degree_sums, radius)


def _variation_bound_from_row_degree_sums(
    row_degree_sums: tuple[
        tuple[Fraction, Fraction, Fraction], ...
    ],
    radius: Fraction,
) -> Fraction:
    return max(
        sum(
            (
                coefficient_sum * radius**degree
                for degree, coefficient_sum in enumerate(
                    degree_sums,
                    start=1,
                )
            ),
            start=Fraction(0),
        )
        for degree_sums in row_degree_sums
    )


def exact_stability_radius_table(
    design: ObservationDesign,
) -> tuple[ExactRadiusEntry, ...]:
    """Evaluate the prescribed half-margin test at every exact radius."""
    exact_design = _validate_pairs(design, required_size=8)
    _, _, inverse = certify_exact_design(exact_design)
    inverse_norm = exact_matrix_infinity_norm(inverse)
    row_degree_sums = exact_jacobian_row_degree_sums(exact_design)
    table: list[ExactRadiusEntry] = []
    for radius in _STABILITY_RADII:
        delta = _variation_bound_from_row_degree_sums(
            row_degree_sums,
            radius,
        )
        product_bound = inverse_norm * delta
        table.append(
            (radius, delta, product_bound, product_bound <= Fraction(1, 2))
        )
    return tuple(table)


def exact_local_stability_constants(
    design: ObservationDesign,
) -> tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
]:
    """Return ``R, Delta, K*Delta, c_R, C_R, K`` for the largest pass."""
    exact_design = _validate_pairs(design, required_size=8)
    _, _, inverse = certify_exact_design(exact_design)
    inverse_norm = exact_matrix_infinity_norm(inverse)
    passing = [
        entry
        for entry in exact_stability_radius_table(exact_design)
        if entry[3]
    ]
    if not passing:
        raise RuntimeError(
            "no prescribed exact radius satisfies the half-margin."
        )
    radius, delta, product_bound, _ = passing[0]
    lower_lipschitz = Fraction(1, inverse_norm) - delta
    if lower_lipschitz <= 0:
        raise RuntimeError(
            "the selected exact lower Lipschitz constant is not positive."
        )
    inverse_lipschitz = Fraction(1, lower_lipschitz)
    return (
        radius,
        delta,
        product_bound,
        lower_lipschitz,
        inverse_lipschitz,
        inverse_norm,
    )


def exact_common_scale_inverse(
    base_inverse: ExactMinor,
    multiplier: Fraction,
) -> ExactMinor:
    """Return ``diag(h^-degree) B(1)^-1`` in deterministic Hall order."""
    row_count, column_count = _matrix_shape(base_inverse)
    if row_count != 8 or column_count != 8:
        raise ValueError(
            "the common-scale inverse requires an eight-by-eight matrix."
        )
    _require_fraction(multiplier, name="multiplier")
    if multiplier == 0:
        raise ValueError("the common scale multiplier must be nonzero.")
    return tuple(
        tuple(multiplier ** (-degree) * value for value in row)
        for degree, row in zip(
            hall_basis_degrees(),
            base_inverse,
            strict=True,
        )
    )


def exact_common_scale_inverse_norm(
    base_inverse: ExactMinor,
    multiplier: Fraction,
) -> Fraction:
    """Return ``max_i |h|^-d_i s_i`` exactly."""
    row_sums = exact_inverse_row_sums(base_inverse)
    _require_fraction(multiplier, name="multiplier")
    if multiplier == 0:
        raise ValueError("the common scale multiplier must be nonzero.")
    return max(
        abs(multiplier) ** (-degree) * row_sum
        for degree, row_sum in zip(
            hall_basis_degrees(),
            row_sums,
            strict=True,
        )
    )


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
    "exact_centered_observation_polynomials",
    "exact_centered_truncated_observation_map",
    "exact_common_scale_inverse",
    "exact_common_scale_inverse_norm",
    "exact_inverse_row_sums",
    "exact_jacobian_difference_polynomials",
    "exact_jacobian_row_degree_sums",
    "exact_jacobian_variation_bound",
    "exact_joint_state_scale_matrix",
    "exact_local_stability_constants",
    "exact_matrix_infinity_norm",
    "exact_matrix_rank",
    "exact_stability_radius_table",
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
    "stability_test_radii",
]

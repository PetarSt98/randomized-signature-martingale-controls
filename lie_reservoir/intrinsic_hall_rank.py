"""Exact finite checks of the fixed-zero Hall-rank grading witness.

The theorem is in research_output/intrinsic_hall_rank_theorem.md.
This bounded harness computes actual shallow-tanh jets, not the triangular
matrix predicted by that proof. Genericity and local truncated recovery
come from the proof; these finite examples give no conditioning guarantee
or exact recovery from a full CDE endpoint.

Reuse the existing Lyndon basis, tensor bracket expansion, rational matrix
operations, and tanh coefficient recurrence. Only the multivariate state
jets are new. A polynomial key is a sorted tuple of coordinate indices,
with repetition: (0, 0, 2) denotes x_0**2 * x_2, not a tensor word.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import TypeAlias

from .certificates import ExactMinor, _tanh_jet
from .free_lie import (
    Bracket,
    Generator,
    LieNode,
    Word,
    lyndon_basis,
    witt_dimension,
    word_expansion,
)


MAX_INPUT_DIM = 3
MAX_DEGREE = 5
MAX_STATE_DIM = 40
MAX_WORDS_AT_TOP_DEGREE = 128

ExactVector: TypeAlias = tuple[Fraction, ...]
StatePolynomial: TypeAlias = dict[tuple[int, ...], Fraction]
FieldJet: TypeAlias = tuple[StatePolynomial, ...]


@dataclass(frozen=True)
class GradingWitness:
    """Exact quotient/padding parameters; every bias is a multiple of q.

    q = atanh(1/2) is never evaluated numerically. State coordinates
    are retained Hall coordinates followed by any inactive coordinates.
    Inactive grading entries are zero; inversion of the grading operator
    is performed only on the positively graded active block.
    """

    input_dim: int
    degree: int
    state_dim: int
    basis: tuple[LieNode, ...]
    homogeneous_dimensions: tuple[int, ...]
    hall_degrees: tuple[int, ...]
    retained_columns: tuple[int, ...]
    state_degrees: tuple[int, ...]
    grading: ExactVector
    adjoints: tuple[ExactMinor, ...]
    lambdas: tuple[ExactMinor, ...]
    weights: tuple[ExactMinor, ...]
    bias_q_multipliers: tuple[ExactVector, ...]
    projected_generators: tuple[ExactVector, ...]

    @property
    def nu(self) -> int:
        return len(self.basis)


def _positive_integer(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")


def tanh_coefficients(value: Fraction, order: int) -> ExactVector:
    """Return tanh(b+t) coefficients through order, given exact tanh(b).

    The existing recurrence (n+1)*a[n+1] = [t**n](1-a(t)**2)
    produces normalized Taylor coefficients, including a[0]. The order
    is any nonnegative integer; witness degrees have separate explicit
    resource limits.
    """
    if not isinstance(value, Fraction):
        raise TypeError("the activation value must be a Fraction.")
    if not -1 < value < 1:
        raise ValueError("a finite real tanh bias requires -1 < value < 1.")
    return _tanh_jet(value, Fraction(1), order)


@cache
def _hall_data(
    input_dim: int, degree: int
) -> tuple[tuple[LieNode, ...], tuple[int, ...], tuple[int, ...], tuple[ExactMinor, ...]]:
    """Derive adjoint columns in the existing tensor/Hall coordinates."""
    levels = tuple(lyndon_basis(input_dim, m) for m in range(1, degree + 1))
    dimensions = tuple(witt_dimension(input_dim, m) for m in range(1, degree + 1))
    if tuple(map(len, levels)) != dimensions:
        raise ArithmeticError("Lyndon basis dimensions disagree with Witt's formula.")
    basis = tuple(node for level in levels for node in level)
    degrees = tuple(m for m, level in enumerate(levels, 1) for _ in level)
    expansions = tuple(dict(word_expansion(node)) for node in basis)
    blocks = []
    offset = 0
    for size in dimensions:
        block = expansions[offset : offset + size]
        leading_words = tuple(min(expansion) for expansion in block)
        if leading_words != tuple(sorted(set(leading_words))):
            raise ArithmeticError("Hall expansions have inconsistent leading words.")
        blocks.append((offset, block))
        offset += size

    adjoints = []
    for letter in range(input_dim):
        matrix = [[Fraction(0) for _ in basis] for _ in basis]
        for column, (node, level) in enumerate(zip(basis, degrees, strict=True)):
            if level == degree:
                continue  # The affine Lie algebra uses the step-degree quotient.
            tensor = dict(word_expansion(Bracket(Generator(letter), node)))
            start, block = blocks[level]
            # Leading-word elimination uses the existing Lyndon expansions.
            # The old scalar inverse utility is capped at size eight, whereas
            # the (3,4) homogeneous block has 18 columns. Keep its API frozen.
            residual = {word: Fraction(value) for word, value in tensor.items()}
            for local, expansion in enumerate(block):
                leading = min(expansion)
                coordinate = residual.get(leading, Fraction(0)) / expansion[leading]
                matrix[start + local][column] = coordinate
                for word, coefficient in expansion.items():
                    residual[word] = residual.get(word, Fraction(0)) - coordinate * coefficient
            # Check every tensor word, not just leading coordinates.
            if any(residual.values()):
                raise ArithmeticError("an adjoint column failed full tensor reconstruction.")
        adjoints.append(tuple(tuple(row) for row in matrix))
    return basis, dimensions, degrees, tuple(adjoints)


def grading_witness(
    input_dim: int, degree: int, state_dim: int | None = None
) -> GradingWitness:
    """Construct the graded quotient or inactive embedding exactly.

    Supported: d<=3, L<=5, d**L<=128, D<=40, all positive integers.
    These are computational resource bounds, not bounds on the theorem.
    For D<nu the first D Hall coordinates retain complete lower grades
    and a prefix of the last retained grade. Before inducing matrices,
    invariance of the discarded ideal under every generator is checked.
    """
    _positive_integer("input_dim", input_dim, MAX_INPUT_DIM)
    _positive_integer("degree", degree, MAX_DEGREE)
    if input_dim**degree > MAX_WORDS_AT_TOP_DEGREE:
        raise ValueError("input_dim**degree exceeds the exact harness word cap.")
    basis, dimensions, degrees, full_adjoints = _hall_data(input_dim, degree)
    if state_dim is None:
        state_dim = len(basis)
    _positive_integer("state_dim", state_dim, MAX_STATE_DIM)
    active = min(state_dim, len(basis))
    retained = tuple(range(active))
    if active < len(basis):
        top = degrees[active - 1]
        if any(level < top for level in degrees[active:]):
            raise ArithmeticError("the retained coordinates omit a lower grade.")
        # Generator invariance suffices because their iterated brackets
        # generate the Lie algebra. This precedes forming quotient matrices.
        if any(
            matrix[row][column]
            for matrix in full_adjoints
            for row in retained
            for column in range(active, len(basis))
        ):
            raise ArithmeticError("the discarded subspace is not a Lie ideal.")
    adjoints = tuple(
        tuple(
            tuple(
                matrix[row][column] if row < active and column < active else Fraction(0)
                for column in range(state_dim)
            )
            for row in range(state_dim)
        )
        for matrix in full_adjoints
    )
    state_degrees = degrees[:active] + (0,) * (state_dim - active)
    lambdas = tuple(
        tuple(
            tuple(
                matrix[row][column] * Fraction(state_degrees[column], state_degrees[row])
                if row < active and column < active else Fraction(0)
                for column in range(state_dim)
            )
            for row in range(state_dim)
        )
        for matrix in adjoints
    )
    projected = tuple(
        tuple(Fraction(int(row == letter and row < active)) for row in range(state_dim))
        for letter in range(input_dim)
    )
    return GradingWitness(
        input_dim=input_dim,
        degree=degree,
        state_dim=state_dim,
        basis=basis,
        homogeneous_dimensions=dimensions,
        hall_degrees=degrees,
        retained_columns=retained,
        state_degrees=state_degrees,
        grading=tuple(map(Fraction, state_degrees)),
        adjoints=adjoints,
        lambdas=lambdas,
        weights=tuple(
            tuple(tuple(-entry for entry in row) for row in matrix)
            for matrix in lambdas
        ),
        bias_q_multipliers=projected,
        projected_generators=projected,
    )


def _multiply_polynomials(
    left: StatePolynomial, right: StatePolynomial, order: int
) -> StatePolynomial:
    result: StatePolynomial = {}
    for a, av in left.items():
        for b, bv in right.items():
            if len(a) + len(b) <= order:
                key = tuple(sorted(a + b))
                result[key] = result.get(key, Fraction(0)) + av * bv
    return {key: value for key, value in result.items() if value}


def exact_tanh_jets(witness: GradingWitness) -> tuple[FieldJet, ...]:
    """Evaluate the actual witness's ordinary state jets through L-1.

    Length-m word responses at zero use at most m-1 generator derivatives.
    Substitute each exact affine row into the recursively generated Taylor
    coefficients. No theorem-predicted Hall entries enter this calculation.
    """
    order = witness.degree - 1
    fields = []
    for matrix, biases in zip(witness.weights, witness.bias_q_multipliers, strict=True):
        field = []
        for row, bias in zip(matrix, biases, strict=True):
            if bias not in (0, 1):
                raise ValueError("the witness requires biases 0 or q.")
            coefficients = tanh_coefficients(bias / 2, order)
            linear = {(index,): value for index, value in enumerate(row) if value}
            power: StatePolynomial = {(): Fraction(1)}
            polynomial: StatePolynomial = {}
            for n, coefficient in enumerate(coefficients):
                for key, value in power.items():
                    polynomial[key] = polynomial.get(key, Fraction(0)) + coefficient * value
                if n < order:
                    power = _multiply_polynomials(power, linear, order)
            field.append({key: value for key, value in polynomial.items() if value})
        fields.append(tuple(field))
    return tuple(fields)


def _directional_derivative(
    polynomial: StatePolynomial, field: FieldJet, order: int
) -> StatePolynomial:
    result: StatePolynomial = {}
    for monomial, coefficient in polynomial.items():
        for index in set(monomial):
            position = monomial.index(index)
            differentiated = monomial[:position] + monomial[position + 1 :]
            factor = coefficient * monomial.count(index)
            for other, value in field[index].items():
                if len(differentiated) + len(other) <= order:
                    key = tuple(sorted(differentiated + other))
                    result[key] = result.get(key, Fraction(0)) + factor * value
    return {key: value for key, value in result.items() if value}


def evaluate_hall_matrix(witness: GradingWitness) -> ExactMinor:
    """Compute all original Hall columns from actual tanh word-response jets.

    Recursion follows F_iv = D F_v V_i (first letter outermost).
    At word length m retain ordinary polynomial degree L-m: at most L-m
    outer derivatives remain. Discarded higher powers cannot contribute
    to the final value at zero. This is finite jet calculus, not an
    assertion that the nonlinear action factors through a Lie quotient.
    """
    fields = exact_tanh_jets(witness)

    @cache
    def word_jet(word: Word) -> FieldJet:
        if len(word) == 1:
            return fields[word[0]]
        inner = word_jet(word[1:])
        return tuple(
            _directional_derivative(component, fields[word[0]], witness.degree - len(word))
            for component in inner
        )

    matrix = [[Fraction(0) for _ in witness.basis] for _ in range(witness.state_dim)]
    for column, node in enumerate(witness.basis):
        for word, coefficient in word_expansion(node):
            for row, polynomial in enumerate(word_jet(word)):
                matrix[row][column] += coefficient * polynomial.get((), Fraction(0))
    return tuple(tuple(row) for row in matrix)


def witness_minor(witness: GradingWitness, matrix: ExactMinor) -> ExactMinor:
    """Select retained Hall columns and active rows from an evaluated matrix."""
    if len(matrix) != witness.state_dim or any(len(row) != witness.nu for row in matrix):
        raise ValueError("matrix shape does not match the witness.")
    return tuple(
        tuple(matrix[row][column] for column in witness.retained_columns)
        for row in range(len(witness.retained_columns))
    )


def zero_response_jacobian(witness: GradingWitness, scale: Fraction) -> ExactMinor:
    """Derivative in Hall coordinates of the truncated polynomial at zero.

    The degree scales multiply every column, including at the degenerate
    scale zero, where the entire derivative vanishes. Nonzero scale and
    full column rank are needed for the proof's local inverse.
    """
    if not isinstance(scale, Fraction):
        raise TypeError("scale must be a Fraction.")
    return tuple(
        tuple(
            value * scale**degree
            for value, degree in zip(row, witness.hall_degrees, strict=True)
        )
        for row in evaluate_hall_matrix(witness)
    )

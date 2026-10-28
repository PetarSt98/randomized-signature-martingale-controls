"""Small exact checks for the truncated tensor-space linear witness.

The certifying paths in this module use only integers and ``Fraction`` values.
The optional SymPy calculation checks two coefficients of the normalized tanh
limit; it does not certify any nonzero value of epsilon.
"""

from __future__ import annotations

from functools import cache
from fractions import Fraction
from itertools import product
from math import factorial, prod
from typing import Mapping, TypeAlias

from .finite_observation import exact_matrix_rank
from .free_lie import Bracket, Generator, LieNode, Word, lyndon_basis, word_expansion
from .graded_reconstruction import truncated_tensor_product


IntegerVector: TypeAlias = tuple[int, ...]
ExactVector: TypeAlias = tuple[Fraction, ...]
IntegerMatrix: TypeAlias = tuple[tuple[int, ...], ...]
ExactMatrix: TypeAlias = tuple[tuple[Fraction, ...], ...]
RightAction: TypeAlias = tuple[int | None, ...]
SparseTensor: TypeAlias = dict[Word, Fraction]

STRUCTURAL_CASES = ((2, 2), (2, 3), (2, 4), (3, 2))
BLOCK_EMBEDDING_CASES = ((2, 2), (2, 3))
TANH_LIMIT_DIAGNOSTIC_CASES = ((2, 2), (2, 3))
MAX_REPRODUCER_STATE_DIM = 40
NORMALIZED_TANH_LIMIT_CERTIFIES_NONZERO_EPSILON = False


def _validate_case(input_dim: int, degree: int) -> None:
    if isinstance(input_dim, bool) or not isinstance(input_dim, int):
        raise TypeError("input_dim must be an integer.")
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError("degree must be an integer.")
    if (input_dim, degree) not in STRUCTURAL_CASES:
        raise ValueError(f"the exact harness is restricted to {STRUCTURAL_CASES}.")


def tensor_state_dimension(input_dim: int, degree: int) -> int:
    """Return ``sum(input_dim**m, m=0..degree)`` in the fixed scope."""
    _validate_case(input_dim, degree)
    return sum(input_dim**level for level in range(degree + 1))


@cache
def tensor_word_basis(input_dim: int, degree: int) -> tuple[Word, ...]:
    """Return the empty word followed by degree-first lexicographic words."""
    _validate_case(input_dim, degree)
    return ((),) + tuple(
        word
        for level in range(1, degree + 1)
        for word in product(range(input_dim), repeat=level)
    )


def nonempty_tensor_words(input_dim: int, degree: int) -> tuple[Word, ...]:
    """Return all nonempty words in the fixed deterministic order."""
    return tensor_word_basis(input_dim, degree)[1:]


@cache
def right_concatenation_actions(
    input_dim: int,
    degree: int,
) -> tuple[RightAction, ...]:
    """Return sparse column actions for ``R_i e_w = e_{w+i}``."""
    basis = tensor_word_basis(input_dim, degree)
    indices = {word: index for index, word in enumerate(basis)}
    return tuple(
        tuple(
            indices[word + (letter,)] if len(word) < degree else None
            for word in basis
        )
        for letter in range(input_dim)
    )


def apply_right_action(
    action: RightAction,
    vector: IntegerVector | ExactVector,
) -> IntegerVector | ExactVector:
    """Apply one sparse right-concatenation action exactly."""
    if len(action) != len(vector):
        raise ValueError("the action and vector dimensions do not agree.")
    result = [0] * len(vector)
    for source, target in enumerate(action):
        if target is not None:
            result[target] += vector[source]
    return tuple(result)


def _integer_identity(size: int) -> IntegerMatrix:
    return tuple(
        tuple(int(row == column) for column in range(size))
        for row in range(size)
    )


def _right_multiply_action(
    matrix: IntegerMatrix,
    action: RightAction,
) -> IntegerMatrix:
    """Return ``matrix @ R`` from the sparse columns of ``R``."""
    if len(matrix) != len(action) or any(len(row) != len(action) for row in matrix):
        raise ValueError("a word-field matrix must agree with the action size.")
    return tuple(
        tuple(0 if target is None else row[target] for target in action)
        for row in matrix
    )


def _integer_matrix_product(
    left: IntegerMatrix,
    right: IntegerMatrix,
) -> IntegerMatrix:
    size = len(left)
    if (
        size == 0
        or len(right) != size
        or any(len(row) != size for row in left)
        or any(len(row) != size for row in right)
    ):
        raise ValueError("integer matrix products require equal square sizes.")
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _integer_matrix_difference(
    left: IntegerMatrix,
    right: IntegerMatrix,
) -> IntegerMatrix:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right, strict=True)
    ):
        raise ValueError("integer matrix dimensions do not agree.")
    return tuple(
        tuple(
            left_value - right_value
            for left_value, right_value in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _integer_matrix_linear_combination(
    terms: tuple[tuple[IntegerMatrix, int], ...],
    size: int,
) -> IntegerMatrix:
    return tuple(
        tuple(
            sum(coefficient * matrix[row][column] for matrix, coefficient in terms)
            for column in range(size)
        )
        for row in range(size)
    )


def _integer_matrix_vector_product(
    matrix: IntegerMatrix,
    vector: IntegerVector,
) -> IntegerVector:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix and vector dimensions do not agree.")
    return tuple(
        sum(entry * value for entry, value in zip(row, vector, strict=True))
        for row in matrix
    )


@cache
def gamma_linear_word_matrix(
    word: Word,
    actions: tuple[RightAction, ...],
) -> IntegerMatrix:
    """Return the matrix of ``Gamma(word) Id`` for the linear fields."""
    if not word:
        raise ValueError("the word must be nonempty.")
    size = len(actions[0])
    matrix = _integer_identity(size)
    # The suffix operator acts first: Gamma(i1...im) Id = R_im ... R_i1 x.
    for letter in reversed(word):
        matrix = _right_multiply_action(matrix, actions[letter])
    return matrix


def gamma_linear_word_evaluation(
    word: Word,
    actions: tuple[RightAction, ...],
    initial_state: IntegerVector | None = None,
) -> IntegerVector:
    """Evaluate ``Gamma(word) Id`` by chronological right concatenation."""
    if not word:
        raise ValueError("the word must be nonempty.")
    state: IntegerVector | ExactVector
    state = (
        (1,) + (0,) * (len(actions[0]) - 1)
        if initial_state is None
        else initial_state
    )
    for letter in word:
        state = apply_right_action(actions[letter], state)
    return tuple(int(value) for value in state)


def word_evaluation_matrix(input_dim: int, degree: int) -> IntegerMatrix:
    """Return the nonempty-coordinate matrix with chronological word columns."""
    words = nonempty_tensor_words(input_dim, degree)
    actions = right_concatenation_actions(input_dim, degree)
    columns = tuple(
        gamma_linear_word_evaluation(word, actions)
        for word in words
    )
    return tuple(
        tuple(column[row] for column in columns)
        for row in range(1, tensor_state_dimension(input_dim, degree))
    )


def length_cutoff_holds(input_dim: int, degree: int) -> bool:
    """Whether every length-``degree+1`` linear word evaluation is zero."""
    actions = right_concatenation_actions(input_dim, degree)
    zero = (0,) * tensor_state_dimension(input_dim, degree)
    return all(
        gamma_linear_word_evaluation(word, actions) == zero
        for word in product(range(input_dim), repeat=degree + 1)
    )


def deterministic_segment_increments(
    input_dim: int,
    degree: int,
) -> tuple[ExactVector, ExactVector]:
    """Return fixed nonzero rational increments for the exact convention check."""
    _validate_case(input_dim, degree)
    first = (Fraction(1, 2), Fraction(-2, 3), Fraction(3, 5))
    second = (Fraction(-1, 3), Fraction(4, 5), Fraction(-2, 7))
    return first[:input_dim], second[:input_dim]


def _segment_signature_tensor(
    increment: ExactVector,
    input_dim: int,
    degree: int,
) -> SparseTensor:
    if len(increment) != input_dim or any(
        not isinstance(value, Fraction) for value in increment
    ):
        raise TypeError("increments must be input_dim-tuples of Fractions.")
    return {
        word: prod(
            (increment[letter] for letter in word),
            start=Fraction(1),
        )
        / Fraction(factorial(len(word)))
        for word in tensor_word_basis(input_dim, degree)
    }


def segment_signature(
    increment: ExactVector,
    input_dim: int,
    degree: int,
) -> ExactVector:
    """Return ``sum_m increment**tensor(m)/m!`` in word coordinates."""
    tensor = _segment_signature_tensor(increment, input_dim, degree)
    return tuple(tensor[word] for word in tensor_word_basis(input_dim, degree))


def _exact_chen_product(
    left: Mapping[Word, Fraction],
    right: Mapping[Word, Fraction],
    input_dim: int,
    degree: int,
) -> SparseTensor:
    """Delegate exact multiplication to the frozen binary tensor product."""
    _validate_case(input_dim, degree)
    if input_dim == 2:
        product_tensor = truncated_tensor_product(left, right)
        return {
            word: coefficient
            for word, coefficient in product_tensor.items()
            if len(word) <= degree
        }

    # In the sole ternary case degree=2, each target word uses at most two
    # letters. Relabel that target subalphabet to {0, 1}, call the existing
    # exact product, and pull back only the requested coefficient.
    result: SparseTensor = {}
    for target in tensor_word_basis(input_dim, degree):
        alphabet = tuple(sorted(set(target)))
        encoding = {letter: index for index, letter in enumerate(alphabet)}

        def encoded(tensor: Mapping[Word, Fraction]) -> SparseTensor:
            return {
                tuple(encoding[letter] for letter in word): coefficient
                for word, coefficient in tensor.items()
                if all(letter in encoding for letter in word)
            }

        encoded_target = tuple(encoding[letter] for letter in target)
        coefficient = truncated_tensor_product(
            encoded(left),
            encoded(right),
        ).get(encoded_target, Fraction(0))
        if coefficient != 0:
            result[target] = coefficient
    return result


def linear_segment_response(
    increment: ExactVector,
    degree: int,
    actions: tuple[RightAction, ...],
    initial_state: ExactVector | None = None,
) -> ExactVector:
    """Apply the exact truncated exponential of one linear segment."""
    if len(increment) != len(actions) or any(
        not isinstance(value, Fraction) for value in increment
    ):
        raise TypeError("increments must match the actions and contain Fractions.")
    size = len(actions[0])
    term = (
        (Fraction(1),) + (Fraction(0),) * (size - 1)
        if initial_state is None
        else initial_state
    )
    if len(term) != size or any(not isinstance(value, Fraction) for value in term):
        raise TypeError("the initial state must be a matching Fraction tuple.")
    result = list(term)
    for order in range(1, degree + 1):
        next_term = [Fraction(0)] * size
        for letter, coefficient in enumerate(increment):
            acted = apply_right_action(actions[letter], term)
            for index, value in enumerate(acted):
                next_term[index] += coefficient * value
        term = tuple(next_term)
        divisor = Fraction(factorial(order))
        for index, value in enumerate(term):
            result[index] += value / divisor
    return tuple(result)


def signature_convention_checks(input_dim: int, degree: int) -> tuple[bool, bool]:
    """Return exact one-segment and two-segment convention statuses."""
    first, second = deterministic_segment_increments(input_dim, degree)
    actions = right_concatenation_actions(input_dim, degree)
    first_tensor = _segment_signature_tensor(first, input_dim, degree)
    second_tensor = _segment_signature_tensor(second, input_dim, degree)
    first_expected = tuple(
        first_tensor[word] for word in tensor_word_basis(input_dim, degree)
    )
    first_response = linear_segment_response(first, degree, actions)
    chen_tensor = _exact_chen_product(
        first_tensor,
        second_tensor,
        input_dim,
        degree,
    )
    two_expected = tuple(
        chen_tensor.get(word, Fraction(0))
        for word in tensor_word_basis(input_dim, degree)
    )
    two_response = linear_segment_response(
        second,
        degree,
        actions,
        initial_state=first_response,
    )
    return first_response == first_expected, two_response == two_expected


@cache
def hall_basis_through_degree(input_dim: int, degree: int) -> tuple[LieNode, ...]:
    """Return the existing Lyndon/Hall basis in degree-first order."""
    _validate_case(input_dim, degree)
    return tuple(
        node
        for level in range(1, degree + 1)
        for node in lyndon_basis(input_dim=input_dim, degree=level)
    )


@cache
def hall_word_column_matrix(input_dim: int, degree: int) -> ExactMatrix:
    """Return exact Hall columns in the complete nonempty word coordinates."""
    words = nonempty_tensor_words(input_dim, degree)
    expansions = tuple(
        dict(word_expansion(node))
        for node in hall_basis_through_degree(input_dim, degree)
    )
    return tuple(
        tuple(Fraction(expansion.get(word, 0)) for expansion in expansions)
        for word in words
    )


def _action_matrix(action: RightAction) -> IntegerMatrix:
    size = len(action)
    return tuple(
        tuple(int(action[column] == row) for column in range(size))
        for row in range(size)
    )


@cache
def represented_linear_lie_matrix(
    node: LieNode,
    actions: tuple[RightAction, ...],
) -> IntegerMatrix:
    """Represent one existing Hall node using ``[V,W]=DW V-DV W``."""
    if isinstance(node, Generator):
        return _action_matrix(actions[node.index])
    if not isinstance(node, Bracket):
        raise TypeError("node must be an existing Generator or Bracket.")
    left = represented_linear_lie_matrix(node.left, actions)
    right = represented_linear_lie_matrix(node.right, actions)
    return _integer_matrix_difference(
        _integer_matrix_product(right, left),
        _integer_matrix_product(left, right),
    )


@cache
def expanded_linear_lie_matrix(
    node: LieNode,
    actions: tuple[RightAction, ...],
) -> IntegerMatrix:
    """Apply Gamma to the existing associative-word expansion of a Hall node."""
    size = len(actions[0])
    terms = tuple(
        (gamma_linear_word_matrix(word, actions), coefficient)
        for word, coefficient in word_expansion(node)
    )
    return _integer_matrix_linear_combination(terms, size)


def hall_convention_checks(input_dim: int, degree: int) -> tuple[bool, int, int]:
    """Return Hall-to-word equality and the exact Hall-column rank."""
    actions = right_concatenation_actions(input_dim, degree)
    nodes = hall_basis_through_degree(input_dim, degree)
    hall_matrix = hall_word_column_matrix(input_dim, degree)
    operator_equal = True
    for column, node in enumerate(nodes):
        represented = represented_linear_lie_matrix(node, actions)
        expanded = expanded_linear_lie_matrix(node, actions)
        if represented != expanded:
            operator_equal = False
            break
        x0 = (1,) + (0,) * (len(actions[0]) - 1)
        evaluated = _integer_matrix_vector_product(represented, x0)[1:]
        expected = tuple(
            int(value)
            for value in (row[column] for row in hall_matrix)
        )
        if evaluated != expected:
            operator_equal = False
            break
    return operator_equal, exact_matrix_rank(hall_matrix), len(nodes)


def embed_right_actions(
    actions: tuple[RightAction, ...],
    extra_dimensions: int,
) -> tuple[RightAction, ...]:
    """Append inactive source coordinates to every right action."""
    if isinstance(extra_dimensions, bool) or not isinstance(extra_dimensions, int):
        raise TypeError("extra_dimensions must be an integer.")
    if extra_dimensions <= 0:
        raise ValueError("extra_dimensions must be positive.")
    return tuple(action + (None,) * extra_dimensions for action in actions)


def block_embedding_checks(
    input_dim: int,
    degree: int,
    extra_dimensions: int = 2,
) -> tuple[bool, bool, bool]:
    """Check the preserved word minor, inactive coordinates, and Hall rank."""
    _validate_case(input_dim, degree)
    if (input_dim, degree) not in BLOCK_EMBEDDING_CASES:
        raise ValueError(f"block checks are restricted to {BLOCK_EMBEDDING_CASES}.")
    words = nonempty_tensor_words(input_dim, degree)
    original_actions = right_concatenation_actions(input_dim, degree)
    embedded_actions = embed_right_actions(original_actions, extra_dimensions)
    base_size = tensor_state_dimension(input_dim, degree)
    word_matrices = tuple(
        gamma_linear_word_matrix(word, embedded_actions)
        for word in words
    )
    columns = tuple(
        gamma_linear_word_evaluation(word, embedded_actions)
        for word in words
    )
    embedded_minor = tuple(
        tuple(column[row] for column in columns)
        for row in range(1, base_size)
    )
    extra_zero = all(
        all(value == 0 for row in matrix[base_size:] for value in row)
        for matrix in word_matrices
    )

    nodes = hall_basis_through_degree(input_dim, degree)
    embedded_hall = tuple(
        tuple(
            Fraction(
                _integer_matrix_vector_product(
                    represented_linear_lie_matrix(node, embedded_actions),
                    (1,) + (0,) * (base_size + extra_dimensions - 1),
                )[row]
            )
            for node in nodes
        )
        for row in tuple(range(1, base_size))
        + tuple(range(base_size, base_size + extra_dimensions))
    )
    original_rank = exact_matrix_rank(hall_word_column_matrix(input_dim, degree))
    return (
        embedded_minor == word_evaluation_matrix(input_dim, degree),
        extra_zero,
        exact_matrix_rank(embedded_hall) == original_rank,
    )


def _sympy_fraction(value: object) -> Fraction:
    numerator, denominator = value.as_numer_denom()  # type: ignore[attr-defined]
    if not numerator.is_Integer or not denominator.is_Integer:
        raise RuntimeError("a tanh-limit coefficient is not rational.")
    return Fraction(int(numerator), int(denominator))


@cache
def noncertifying_normalized_tanh_limit_coefficients(
    input_dim: int,
    degree: int,
) -> tuple[ExactMatrix, ExactMatrix]:
    """Return the exact epsilon^0 and epsilon^1 normalized matrix coefficients.

    This diagnostic never evaluates a nonzero epsilon and is not a determinant,
    rank, conditioning, or nonvanishing certificate.
    """
    _validate_case(input_dim, degree)
    if (input_dim, degree) not in TANH_LIMIT_DIAGNOSTIC_CASES:
        raise ValueError(
            f"the tanh diagnostic is restricted to {TANH_LIMIT_DIAGNOSTIC_CASES}."
        )
    try:
        import sympy as sp
    except ImportError as error:
        raise RuntimeError("SymPy is unavailable for the optional diagnostic.") from error

    words = nonempty_tensor_words(input_dim, degree)
    actions = right_concatenation_actions(input_dim, degree)
    size = tensor_state_dimension(input_dim, degree)
    epsilon = sp.Symbol("epsilon")
    state_symbols = sp.symbols(f"x0:{size}")
    state = sp.Matrix(state_symbols)
    fields = tuple(
        (sp.Matrix(_action_matrix(action)) * state).applyfunc(
            lambda value: sp.tanh(epsilon * value) / epsilon
        )
        for action in actions
    )

    @cache
    def normalized_word_field(word: Word):
        if len(word) == 1:
            return fields[word[0]]
        inner = normalized_word_field(word[1:])
        return inner.jacobian(state) * fields[word[0]]

    x0_substitution = {
        symbol: int(index == 0)
        for index, symbol in enumerate(state_symbols)
    }
    constant_columns: list[tuple[Fraction, ...]] = []
    linear_columns: list[tuple[Fraction, ...]] = []
    for word in words:
        evaluated = normalized_word_field(word).subs(x0_substitution)
        constant_column: list[Fraction] = []
        linear_column: list[Fraction] = []
        for coordinate in range(1, size):
            series = sp.series(
                evaluated[coordinate],
                epsilon,
                0,
                2,
            ).removeO().expand()
            constant_column.append(_sympy_fraction(series.coeff(epsilon, 0)))
            linear_column.append(_sympy_fraction(series.coeff(epsilon, 1)))
        constant_columns.append(tuple(constant_column))
        linear_columns.append(tuple(linear_column))

    constant_matrix = tuple(
        tuple(column[row] for column in constant_columns)
        for row in range(len(words))
    )
    linear_matrix = tuple(
        tuple(column[row] for column in linear_columns)
        for row in range(len(words))
    )
    return constant_matrix, linear_matrix


__all__ = [
    "BLOCK_EMBEDDING_CASES",
    "MAX_REPRODUCER_STATE_DIM",
    "NORMALIZED_TANH_LIMIT_CERTIFIES_NONZERO_EPSILON",
    "STRUCTURAL_CASES",
    "TANH_LIMIT_DIAGNOSTIC_CASES",
    "apply_right_action",
    "block_embedding_checks",
    "deterministic_segment_increments",
    "embed_right_actions",
    "expanded_linear_lie_matrix",
    "gamma_linear_word_evaluation",
    "gamma_linear_word_matrix",
    "hall_basis_through_degree",
    "hall_convention_checks",
    "hall_word_column_matrix",
    "length_cutoff_holds",
    "linear_segment_response",
    "noncertifying_normalized_tanh_limit_coefficients",
    "nonempty_tensor_words",
    "represented_linear_lie_matrix",
    "right_concatenation_actions",
    "segment_signature",
    "signature_convention_checks",
    "tensor_state_dimension",
    "tensor_word_basis",
    "word_evaluation_matrix",
]

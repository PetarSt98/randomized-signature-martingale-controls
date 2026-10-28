"""Exact homogeneous Lyndon bases with standard Lyndon bracketing.

The implementation works one homogeneous degree at a time. Internal generator
indices are zero-based, while :func:`format_lie_node` uses one-based names for
human-readable output.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TypeAlias


@dataclass(frozen=True)
class Generator:
    """One zero-based free-Lie generator."""

    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("index must be an integer.")
        if self.index < 0:
            raise ValueError("index must be non-negative.")


@dataclass(frozen=True)
class Bracket:
    """A binary Lie bracket."""

    left: LieNode
    right: LieNode

    def __post_init__(self) -> None:
        if not isinstance(self.left, (Generator, Bracket)):
            raise TypeError("left must be a Generator or Bracket.")
        if not isinstance(self.right, (Generator, Bracket)):
            raise TypeError("right must be a Generator or Bracket.")


LieNode: TypeAlias = Generator | Bracket
Word: TypeAlias = tuple[int, ...]
WordExpansion: TypeAlias = tuple[tuple[Word, int], ...]


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _is_lyndon(word: Word) -> bool:
    """Whether ``word`` is strictly smaller than every proper suffix."""
    return bool(word) and all(word < word[split:] for split in range(1, len(word)))


def lyndon_words(input_dim: int, degree: int) -> tuple[Word, ...]:
    """Return all degree-``degree`` Lyndon words in lexicographic order."""
    _validate_positive_integer("input_dim", input_dim)
    _validate_positive_integer("degree", degree)
    words = product(range(input_dim), repeat=degree)
    return tuple(word for word in words if _is_lyndon(word))


def _standard_factorization(word: Word) -> tuple[Word, Word]:
    """Return ``w = uv`` with ``v`` the longest proper Lyndon suffix."""
    if len(word) < 2:
        raise ValueError("standard factorization requires a nontrivial Lyndon word.")
    if not _is_lyndon(word):
        raise ValueError("standard factorization requires a Lyndon word.")

    for split in range(1, len(word)):
        suffix = word[split:]
        if _is_lyndon(suffix):
            return word[:split], suffix
    raise RuntimeError("a nontrivial Lyndon word must have a Lyndon suffix.")


def _standard_bracketing(word: Word) -> LieNode:
    if len(word) == 1:
        return Generator(word[0])
    left_word, right_word = _standard_factorization(word)
    return Bracket(
        left=_standard_bracketing(left_word),
        right=_standard_bracketing(right_word),
    )


def lyndon_basis(input_dim: int, degree: int) -> tuple[LieNode, ...]:
    """Return the standard-bracketed Lyndon basis of one homogeneous degree."""
    return tuple(
        _standard_bracketing(word)
        for word in lyndon_words(input_dim=input_dim, degree=degree)
    )


def word_expansion(node: LieNode) -> WordExpansion:
    """Expand a bracket tree exactly using ``[u, v] = uv - vu``."""
    if isinstance(node, Generator):
        return (((node.index,), 1),)
    if not isinstance(node, Bracket):
        raise TypeError("node must be a Generator or Bracket.")

    coefficients: dict[Word, int] = {}
    for left_word, left_coefficient in word_expansion(node.left):
        for right_word, right_coefficient in word_expansion(node.right):
            coefficient = left_coefficient * right_coefficient
            forward = left_word + right_word
            reverse = right_word + left_word
            coefficients[forward] = coefficients.get(forward, 0) + coefficient
            coefficients[reverse] = coefficients.get(reverse, 0) - coefficient

    return tuple(
        (word, coefficient)
        for word, coefficient in sorted(coefficients.items())
        if coefficient != 0
    )


def _mobius(value: int) -> int:
    """Return the number-theoretic Moebius function for a positive integer."""
    remaining = value
    prime_factor_count = 0
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor != 0:
            divisor += 1
            continue
        remaining //= divisor
        prime_factor_count += 1
        if remaining % divisor == 0:
            return 0
        divisor += 1
    if remaining > 1:
        prime_factor_count += 1
    return -1 if prime_factor_count % 2 else 1


def witt_dimension(input_dim: int, degree: int) -> int:
    """Return the exact homogeneous free-Lie dimension from Witt's formula."""
    _validate_positive_integer("input_dim", input_dim)
    _validate_positive_integer("degree", degree)
    numerator = sum(
        _mobius(divisor) * input_dim ** (degree // divisor)
        for divisor in range(1, degree + 1)
        if degree % divisor == 0
    )
    dimension, remainder = divmod(numerator, degree)
    if remainder != 0:
        raise ArithmeticError("Witt's formula did not produce an integer.")
    return dimension


def format_lie_node(node: LieNode) -> str:
    """Format generators as ``X1``, ``X2``, ... and brackets recursively."""
    if isinstance(node, Generator):
        return f"X{node.index + 1}"
    if not isinstance(node, Bracket):
        raise TypeError("node must be a Generator or Bracket.")
    return f"[{format_lie_node(node.left)},{format_lie_node(node.right)}]"


__all__ = [
    "Bracket",
    "Generator",
    "LieNode",
    "format_lie_node",
    "lyndon_basis",
    "lyndon_words",
    "witt_dimension",
    "word_expansion",
]

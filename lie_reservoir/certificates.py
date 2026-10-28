"""Exact certificates through homogeneous degree four for one tanh witness.

This module is deliberately narrow.  It reconstructs four selected square
minors with exact :class:`fractions.Fraction` Taylor jets; it is not a general
symbolic-algebra system.

For scalar fields the repository convention is

``[V, W] = W' V - V' W``.

The exact witness uses ``q = atanh(1/2) = log(3) / 2`` and

``V_1(x) = tanh(x)``, ``V_2(x) = tanh(q) = 1/2``

at degree-specific subsets of ``(-q, 0, q)``.  All activation values and
Taylor coefficients needed by the recursive brackets are therefore rational.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Literal, TypeAlias

from .free_lie import Bracket, Generator, LieNode, format_lie_node, lyndon_basis


ExactMinor: TypeAlias = tuple[tuple[Fraction, ...], ...]
CertificateStatus: TypeAlias = Literal[
    "exact_nonzero",
    "taylor_coefficient_nonzero",
    "interval_certified_nonzero",
    "numerical_candidate_only",
    "inconclusive",
]

ALLOWED_CERTIFICATE_STATUSES = (
    "exact_nonzero",
    "taylor_coefficient_nonzero",
    "interval_certified_nonzero",
    "numerical_candidate_only",
    "inconclusive",
)


@dataclass(frozen=True)
class ExactMinorCertificate:
    """Immutable exact data for the selected homogeneous minor."""

    architecture: str
    input_dim: int
    state_dim: int
    degree: int
    basis_labels: tuple[str, ...]
    constant_definition: str
    activation_identities: tuple[str, ...]
    weight_entries: tuple[str, ...]
    bias_entries: tuple[str, ...]
    evaluation_points: tuple[str, ...]
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    exact_minor: ExactMinor
    exact_determinant: Fraction
    certification_method: str
    status: CertificateStatus


@dataclass(frozen=True)
class ExactThroughDegreeFourCertificate:
    """Combined exact certificate for homogeneous degrees one through four."""

    architecture: str
    input_dim: int
    state_dim: int
    degrees: tuple[int, ...]
    homogeneous_certificates: tuple[ExactMinorCertificate, ...]
    exact_determinant_product: Fraction
    certification_method: str
    status: CertificateStatus
    conclusion: str


@dataclass(frozen=True)
class ReportedFloatValue:
    """One explicitly reported conversion used only for a numerical check."""

    value: float
    exact_value: str
    conversion: str
    purpose: str
    warning: str


@dataclass(frozen=True)
class ReportedFloatMinor:
    """Explicitly reported binary64 conversion of an exact minor."""

    values: tuple[tuple[float, ...], ...]
    exact_values: tuple[tuple[str, ...], ...]
    conversion: str
    purpose: str
    warning: str


def _validate_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer.")
    if order < 0:
        raise ValueError("order must be non-negative.")


def _tanh_jet(
    value: Fraction,
    slope: Fraction,
    order: int,
) -> tuple[Fraction, ...]:
    """Return normalized Taylor coefficients using ``y'=slope*(1-y^2)``."""
    _validate_order(order)
    coefficients = [value]
    for power in range(order):
        square_coefficient = sum(
            coefficients[index] * coefficients[power - index]
            for index in range(power + 1)
        )
        constant_coefficient = Fraction(1) if power == 0 else Fraction(0)
        coefficients.append(
            slope
            * (constant_coefficient - square_coefficient)
            / Fraction(power + 1)
        )
    return tuple(coefficients)


def _multiply_series(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
    order: int,
) -> tuple[Fraction, ...]:
    return tuple(
        sum(left[index] * right[power - index] for index in range(power + 1))
        for power in range(order + 1)
    )


def _differentiate_series(
    series: tuple[Fraction, ...],
    order: int,
) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(power + 1) * series[power + 1]
        for power in range(order + 1)
    )


def _bracket_series(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
    order: int,
) -> tuple[Fraction, ...]:
    """Apply exactly ``[left,right] = right' left - left' right``."""
    left_derivative = _differentiate_series(left, order)
    right_derivative = _differentiate_series(right, order)
    forward = _multiply_series(right_derivative, left, order)
    reverse = _multiply_series(left_derivative, right, order)
    return tuple(
        forward[power] - reverse[power]
        for power in range(order + 1)
    )


def _represented_jet(
    node: LieNode,
    activation_values: tuple[Fraction, ...],
    slopes: tuple[Fraction, ...],
    order: int,
) -> tuple[Fraction, ...]:
    if isinstance(node, Generator):
        if node.index >= len(activation_values):
            raise ValueError("generator index exceeds the supplied fields.")
        return _tanh_jet(
            activation_values[node.index],
            slopes[node.index],
            order,
        )
    if not isinstance(node, Bracket):
        raise TypeError("node must be a Generator or Bracket.")
    left = _represented_jet(
        node.left,
        activation_values,
        slopes,
        order + 1,
    )
    right = _represented_jet(
        node.right,
        activation_values,
        slopes,
        order + 1,
    )
    return _bracket_series(left, right, order)


def _determinant(minor: ExactMinor) -> Fraction:
    """Return the exact determinant of a nonempty square Fraction matrix."""
    size = len(minor)
    if size == 0 or any(len(row) != size for row in minor):
        raise ValueError("minor must be a nonempty square matrix.")
    work = [list(row) for row in minor]
    determinant = Fraction(1)
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
            return Fraction(0)
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            determinant = -determinant
        pivot = work[column][column]
        determinant *= pivot
        for row in range(column + 1, size):
            factor = work[row][column] / pivot
            for trailing_column in range(column + 1, size):
                work[row][trailing_column] -= (
                    factor * work[column][trailing_column]
                )
    return determinant


def classify_certificate(
    *,
    exact_determinant: Fraction | int | None,
    numerical_determinant: Real | None = None,
) -> CertificateStatus:
    """Classify evidence without promoting a numerical determinant to exact."""
    if isinstance(exact_determinant, bool) or (
        exact_determinant is not None
        and not isinstance(exact_determinant, (Fraction, int))
    ):
        raise TypeError("exact_determinant must be an int, Fraction, or None.")
    if numerical_determinant is not None:
        if isinstance(numerical_determinant, bool) or not isinstance(
            numerical_determinant,
            Real,
        ):
            raise TypeError("numerical_determinant must be a real number or None.")
        if not math.isfinite(float(numerical_determinant)):
            raise ValueError("numerical_determinant must be finite.")
    if exact_determinant is not None:
        return "exact_nonzero" if exact_determinant != 0 else "inconclusive"
    if numerical_determinant is not None:
        return "numerical_candidate_only"
    return "inconclusive"


def _degree_data(
    degree: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[Fraction, Fraction], ...],
    ExactMinor,
    Fraction,
]:
    """Return immutable expected data used to guard exact reconstruction."""
    half = Fraction(1, 2)
    zero = Fraction(0)
    if degree == 1:
        return (
            ("X1", "X2"),
            ("0", "log(3)/2"),
            ((zero, half), (half, half)),
            ((zero, half), (half, half)),
            Fraction(-1, 4),
        )
    if degree == 2:
        return (
            ("[X1,X2]",),
            ("0",),
            ((zero, half),),
            ((-half,),),
            Fraction(-1, 2),
        )
    if degree == 3:
        return (
            ("[X1,[X1,X2]]", "[[X1,X2],X2]"),
            ("-log(3)/2", "log(3)/2"),
            ((-half, half), (half, half)),
            (
                (Fraction(15, 32), Fraction(3, 16)),
                (Fraction(15, 32), Fraction(-3, 16)),
            ),
            Fraction(-45, 256),
        )
    if degree == 4:
        return (
            (
                "[X1,[X1,[X1,X2]]]",
                "[X1,[[X1,X2],X2]]",
                "[[[X1,X2],X2],X2]",
            ),
            ("-log(3)/2", "0", "log(3)/2"),
            ((-half, half), (zero, half), (half, half)),
            (
                (
                    Fraction(-57, 128),
                    Fraction(-3, 32),
                    Fraction(3, 64),
                ),
                (Fraction(-1, 2), zero, Fraction(1, 4)),
                (
                    Fraction(-57, 128),
                    Fraction(3, 32),
                    Fraction(3, 64),
                ),
            ),
            Fraction(135, 8192),
        )
    raise ValueError("degree must be one of 1, 2, 3, or 4.")


def degree_exact_certificate(degree: int) -> ExactMinorCertificate:
    """Construct one exact complete-basis homogeneous certificate."""
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError("degree must be an integer.")
    (
        expected_labels,
        evaluation_points,
        activation_values,
        expected_minor,
        expected_determinant,
    ) = _degree_data(degree)

    basis = lyndon_basis(input_dim=2, degree=degree)
    basis_labels = tuple(format_lie_node(node) for node in basis)
    if basis_labels != expected_labels:
        raise RuntimeError(
            f"the degree-{degree} Lyndon basis ordering changed."
        )

    slopes = (Fraction(1), Fraction(0))
    exact_minor = tuple(
        tuple(
            _represented_jet(node, point_values, slopes, order=0)[0]
            for node in basis
        )
        for point_values in activation_values
    )
    if exact_minor != expected_minor:
        raise RuntimeError(
            f"the exact degree-{degree} bracket reconstruction changed."
        )
    exact_determinant = _determinant(exact_minor)
    if exact_determinant != expected_determinant:
        raise RuntimeError(
            f"the exact degree-{degree} determinant changed unexpectedly."
        )

    status = classify_certificate(exact_determinant=exact_determinant)
    if status != "exact_nonzero":
        raise RuntimeError("the exact nonzero determinant was not certified.")
    return ExactMinorCertificate(
        architecture="shallow_biased_tanh",
        input_dim=2,
        state_dim=1,
        degree=degree,
        basis_labels=basis_labels,
        constant_definition="q = atanh(1/2) = log(3)/2",
        activation_identities=(
            "exp(2*q) = 3",
            "tanh(q) = (3 - 1)/(3 + 1) = 1/2",
            "tanh(-q) = -1/2",
        ),
        weight_entries=("1", "0"),
        bias_entries=("0", "log(3)/2"),
        evaluation_points=evaluation_points,
        selected_rows=tuple(range(len(exact_minor))),
        selected_columns=tuple(range(len(basis))),
        exact_minor=exact_minor,
        exact_determinant=exact_determinant,
        certification_method="exact_symbolic_determinant_simplification",
        status=status,
    )


def degree_four_exact_certificate() -> ExactMinorCertificate:
    """Construct the preserved exact complete-basis degree-four certificate."""
    return degree_exact_certificate(degree=4)


def through_degree_four_exact_certificate() -> ExactThroughDegreeFourCertificate:
    """Construct the combined exact homogeneous degree-one-to-four result."""
    degrees = (1, 2, 3, 4)
    certificates = tuple(
        degree_exact_certificate(degree=degree)
        for degree in degrees
    )
    if any(certificate.status != "exact_nonzero" for certificate in certificates):
        raise RuntimeError("a homogeneous determinant was not certified.")
    determinant_product = Fraction(1)
    for certificate in certificates:
        determinant_product *= certificate.exact_determinant
    expected_product = Fraction(-6075, 16777216)
    if determinant_product != expected_product:
        raise RuntimeError("the combined exact determinant product changed.")
    status = classify_certificate(exact_determinant=determinant_product)
    if status != "exact_nonzero":
        raise RuntimeError("the combined determinant was not certified.")
    return ExactThroughDegreeFourCertificate(
        architecture="shallow_biased_tanh",
        input_dim=2,
        state_dim=1,
        degrees=degrees,
        homogeneous_certificates=certificates,
        exact_determinant_product=determinant_product,
        certification_method="exact_symbolic_determinant_simplification",
        status=status,
        conclusion=(
            "the representation is injective on each homogeneous free-Lie "
            "component L_{2,m}, m=1,2,3,4"
        ),
    )


def fraction_string(value: Fraction) -> str:
    """Serialize a fraction without any floating-point conversion."""
    if not isinstance(value, Fraction):
        raise TypeError("value must be a Fraction.")
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def report_fraction_as_float(
    value: Fraction,
    *,
    purpose: str,
) -> ReportedFloatValue:
    """Convert one exact value to binary64 with an explicit audit record."""
    if not isinstance(value, Fraction):
        raise TypeError("value must be a Fraction.")
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be a non-empty string.")
    return ReportedFloatValue(
        value=float(value),
        exact_value=fraction_string(value),
        conversion="Fraction -> Python binary64 float",
        purpose=purpose,
        warning="This conversion is a numerical cross-check, not certification.",
    )


def report_minor_as_float(
    minor: ExactMinor,
    *,
    purpose: str,
) -> ReportedFloatMinor:
    """Convert an exact minor to binary64 with an explicit audit record."""
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be a non-empty string.")
    exact_values = tuple(
        tuple(fraction_string(value) for value in row)
        for row in minor
    )
    values = tuple(
        tuple(float(value) for value in row)
        for row in minor
    )
    return ReportedFloatMinor(
        values=values,
        exact_values=exact_values,
        conversion="Fraction matrix -> Python binary64 float matrix",
        purpose=purpose,
        warning="This conversion is a numerical cross-check, not certification.",
    )


__all__ = [
    "ALLOWED_CERTIFICATE_STATUSES",
    "CertificateStatus",
    "ExactMinor",
    "ExactMinorCertificate",
    "ExactThroughDegreeFourCertificate",
    "ReportedFloatMinor",
    "ReportedFloatValue",
    "classify_certificate",
    "degree_exact_certificate",
    "degree_four_exact_certificate",
    "fraction_string",
    "report_fraction_as_float",
    "report_minor_as_float",
    "through_degree_four_exact_certificate",
]

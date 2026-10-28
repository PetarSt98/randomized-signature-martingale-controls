"""Reproduce the exact local-stability certificate without writing files."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lie_reservoir import fraction_string  # noqa: E402
from lie_reservoir.finite_observation import (  # noqa: E402
    certify_exact_design,
    common_rescale_design,
    exact_centered_observation_polynomials,
    exact_common_scale_inverse,
    exact_common_scale_inverse_norm,
    exact_inverse_row_sums,
    exact_jacobian_difference_polynomials,
    exact_jacobian_row_degree_sums,
    exact_local_stability_constants,
    exact_matrix_infinity_norm,
    exact_stability_radius_table,
    hall_basis_degrees,
    hall_basis_labels,
)
from research_output.reproduce_graded_reconstruction import (  # noqa: E402
    _semantic_json_sha256,
)


F = Fraction
ARTIFACT = ROOT / "research_output" / "local_stability_certificate.json"
FINITE_ARTIFACT = (
    ROOT / "research_output" / "finite_observation_certificate.json"
)
COMMON_MULTIPLIERS = (F(2), F(-2), F(1, 2), F(-1, 2))


def _fraction(value: str) -> Fraction:
    if not isinstance(value, str):
        raise RuntimeError("exact certificate values must be strings.")
    try:
        return F(value)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError("an exact certificate fraction is invalid.") from error


def _fraction_vector(values: tuple[Fraction, ...]) -> list[str]:
    return [fraction_string(value) for value in values]


def _fraction_matrix(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> list[list[str]]:
    return [_fraction_vector(row) for row in matrix]


def _parse_fraction_matrix(values: Any) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(values, list):
        raise RuntimeError("an exact certificate matrix must be a list.")
    try:
        return tuple(tuple(_fraction(value) for value in row) for row in values)
    except TypeError as error:
        raise RuntimeError("an exact certificate matrix row is invalid.") from error


def _design_payload(design) -> list[dict[str, str]]:
    return [
        {"state": state_label, "scale": fraction_string(scale)}
        for state_label, scale in design
    ]


def _parse_design(values: Any):
    if not isinstance(values, list):
        raise RuntimeError("the selected design must be a list.")
    try:
        return tuple(
            (entry["state"], _fraction(entry["scale"])) for entry in values
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("a selected design entry is invalid.") from error


def _polynomial_payload(polynomial) -> dict[str, str]:
    return {
        ",".join(str(exponent) for exponent in multiindex): fraction_string(
            coefficient
        )
        for multiindex, coefficient in polynomial.items()
    }


def _polynomial_matrix_payload(polynomial_matrix) -> list[list[Any]]:
    return [
        [_polynomial_payload(polynomial) for polynomial in row]
        for row in polynomial_matrix
    ]


def _read_finite_certificate() -> dict[str, Any]:
    with FINITE_ARTIFACT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("the finite-observation certificate is invalid.")
    return payload


def _validated_finite_certificate(
    payload: dict[str, Any] | None = None,
):
    """Return the fixed Gate-3 design and exact linear algebra after checks."""
    finite_payload = (
        deepcopy(payload) if payload is not None else _read_finite_certificate()
    )
    if finite_payload.get("status") != "PROVED":
        raise RuntimeError("the finite-observation theorem is not certified.")

    stored_hall = finite_payload.get("hall_basis")
    expected_hall = [
        {"label": label, "degree": degree}
        for label, degree in zip(
            hall_basis_labels(),
            hall_basis_degrees(),
            strict=True,
        )
    ]
    if stored_hall != expected_hall:
        raise RuntimeError("the certified Hall ordering is inconsistent.")

    design = _parse_design(finite_payload.get("selected_design"))
    matrix, determinant, inverse = certify_exact_design(design)
    zero_jacobian = finite_payload.get("zero_jacobian")
    if not isinstance(zero_jacobian, dict):
        raise RuntimeError("the certified zero Jacobian is missing.")
    if zero_jacobian.get("hall_column_order") != list(hall_basis_labels()):
        raise RuntimeError("the certified Jacobian column order is inconsistent.")
    if _parse_fraction_matrix(zero_jacobian.get("exact_matrix")) != matrix:
        raise RuntimeError("the certified Jacobian matrix is inconsistent.")
    if _fraction(zero_jacobian.get("exact_determinant")) != determinant:
        raise RuntimeError("the certified Jacobian determinant is inconsistent.")
    if _parse_fraction_matrix(zero_jacobian.get("exact_inverse")) != inverse:
        raise RuntimeError("the certified Jacobian inverse is inconsistent.")
    return design, matrix, determinant, inverse


def _scaled_column_matrix(matrix, multiplier: Fraction):
    degrees = hall_basis_degrees()
    return tuple(
        tuple(
            value * multiplier**degrees[column]
            for column, value in enumerate(row)
        )
        for row in matrix
    )


def build_certificate_payload() -> dict[str, Any]:
    """Recompute every exact Gate-4 certificate field."""
    design, matrix, determinant, inverse = _validated_finite_certificate()
    response_polynomials = exact_centered_observation_polynomials(design)
    jacobian_polynomials = exact_jacobian_difference_polynomials(design)
    row_degree_sums = exact_jacobian_row_degree_sums(design)
    inverse_row_sums = exact_inverse_row_sums(inverse)
    inverse_norm = exact_matrix_infinity_norm(inverse)
    radius_table = exact_stability_radius_table(design)
    (
        radius,
        delta,
        product_bound,
        lower_lipschitz,
        inverse_lipschitz,
        computed_inverse_norm,
    ) = exact_local_stability_constants(design)
    if computed_inverse_norm != inverse_norm:
        raise RuntimeError("the exact inverse norm computations disagree.")
    if product_bound > F(1, 2):
        raise RuntimeError("the selected exact radius fails the half-margin.")
    if next(entry for entry in radius_table if entry[3])[0] != radius:
        raise RuntimeError("the selected exact radius is not the largest pass.")

    simple_inverse_bound = 2 * inverse_norm
    if inverse_lipschitz > simple_inverse_bound:
        raise RuntimeError("the half-margin inverse bound is inconsistent.")

    common_scale_checks: list[dict[str, Any]] = []
    for multiplier in COMMON_MULTIPLIERS:
        scaled_design = common_rescale_design(design, multiplier)
        scaled_matrix, _, scaled_inverse = certify_exact_design(scaled_design)
        expected_matrix = _scaled_column_matrix(matrix, multiplier)
        expected_inverse = exact_common_scale_inverse(inverse, multiplier)
        if scaled_matrix != expected_matrix or scaled_inverse != expected_inverse:
            raise RuntimeError("an exact common-scale factorization failed.")
        direct_norm = exact_matrix_infinity_norm(scaled_inverse)
        formula_norm = exact_common_scale_inverse_norm(inverse, multiplier)
        if direct_norm != formula_norm:
            raise RuntimeError("the exact common-scale row-sum formula failed.")
        common_scale_checks.append(
            {
                "multiplier": fraction_string(multiplier),
                "exact_inverse_norm": fraction_string(direct_norm),
                "jacobian_factorization_verified": True,
                "inverse_factorization_verified": True,
                "row_sum_formula_verified": True,
            }
        )

    union_monomials = set().union(
        *(set(polynomial) for polynomial in response_polynomials)
    )
    radius_rows = [
        {
            "radius": fraction_string(tested_radius),
            "delta": fraction_string(tested_delta),
            "inverse_norm_times_delta": fraction_string(tested_product),
            "half_margin_passes": passes,
        }
        for tested_radius, tested_delta, tested_product, passes in radius_table
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "exact_local_stability_certificate",
        "status": "EXACT_LOCAL_STABILITY",
        "scope": {
            "observation_object": (
                "degree-four truncated finite-scale response map"
            ),
            "hall_coordinate_dimension": 8,
            "scalar_observation_count": 8,
            "vector_norm": "infinity norm",
            "matrix_norm": "induced infinity norm",
            "certified_domain": "closed Hall-coordinate cube about zero",
        },
        "authoritative_finite_observation_certificate": {
            "path": "research_output/finite_observation_certificate.json",
            "semantic_sha256": _semantic_json_sha256(FINITE_ARTIFACT),
        },
        "hall_basis": [
            {"label": label, "degree": degree}
            for label, degree in zip(
                hall_basis_labels(),
                hall_basis_degrees(),
                strict=True,
            )
        ],
        "selected_design": _design_payload(design),
        "zero_jacobian": {
            "exact_matrix": _fraction_matrix(matrix),
            "exact_determinant": fraction_string(determinant),
            "exact_inverse": _fraction_matrix(inverse),
            "exact_inverse_row_sums": _fraction_vector(inverse_row_sums),
            "exact_inverse_infinity_norm": fraction_string(inverse_norm),
        },
        "exact_response_polynomials": {
            "multiindex_order": "eight deterministic Hall coordinates",
            "ordinary_degree_at_most": 4,
            "distinct_nonzero_monomial_count": len(union_monomials),
            "total_nonzero_output_term_count": sum(
                len(polynomial) for polynomial in response_polynomials
            ),
        },
        "exact_jacobian_difference_polynomials": {
            "meaning": "D Phi(z)-D Phi(0)",
            "multiindex_encoding": "comma-separated exponents in Hall order",
            "constant_term_zero": True,
            "entries": _polynomial_matrix_payload(jacobian_polynomials),
            "absolute_row_coefficient_sums_by_degree_1_2_3": [
                _fraction_vector(degree_sums)
                for degree_sums in row_degree_sums
            ],
        },
        "radius_search": {
            "selection_rule": (
                "largest listed radius with ||B^-1||_infinity Delta(R) <= 1/2"
            ),
            "coefficient_sum_majorant": (
                "max row sum of sum |c_alpha| R^|alpha| over entries of "
                "D Phi-B"
            ),
            "complete_exact_table": radius_rows,
            "selected_radius": fraction_string(radius),
            "selected_delta": fraction_string(delta),
            "selected_inverse_norm_times_delta": fraction_string(product_bound),
            "half_margin": "1/2",
        },
        "stability_constants": {
            "exact_lower_lipschitz_constant": fraction_string(lower_lipschitz),
            "exact_inverse_lipschitz_bound": fraction_string(inverse_lipschitz),
            "half_margin_simple_inverse_bound": fraction_string(
                simple_inverse_bound
            ),
            "simple_bound_dominates_exact_bound": True,
        },
        "common_scale_inverse_amplification": {
            "hall_degree_order": list(hall_basis_degrees()),
            "exact_inverse_row_sums": _fraction_vector(inverse_row_sums),
            "jacobian_identity": "B(h)=B(1) diag(h^d_i)",
            "inverse_identity": "B(h)^-1=diag(h^-d_i) B(1)^-1",
            "norm_identity": "||B(h)^-1||_infinity=max_i |h|^-d_i s_i",
            "exact_checks": common_scale_checks,
            "asymptotic": "Theta(|h|^-4) as h tends to zero",
            "degree_four_lower_coefficient": "160/33",
            "uniform_upper_coefficient_for_0_lt_abs_h_le_1": "23812/2151",
            "interpretation": "zero-Jacobian inverse-linearization amplification",
        },
        "theorem_assumptions": [
            "the fixed certified eight-pair Gate-3 design is used",
            "Phi is the exact degree-four truncated polynomial response map",
            "both Hall-coordinate vectors lie in the certified closed cube",
            "the infinity norm and its induced matrix norm are used",
            "the exact coefficient-sum Jacobian majorant is used",
        ],
        "conditional_perturbation_corollary": (
            "if z and z_hat lie in the cube and ||Phi(z_hat)-Phi(z)||_infinity "
            "is at most epsilon, then ||z_hat-z||_infinity is at most "
            "C_R epsilon"
        ),
        "exclusions": [
            "actual full CDE observations",
            "degree-five and higher remainder bounds",
            "bounds on observation noise, solver error, or truncation error",
            "global invertibility",
            "common-scale nonlinear stability",
            "condition-number scaling",
            "scale optimization",
        ],
        "reproduction": {
            "command": "python research_output/reproduce_local_stability.py",
            "writes_files_by_default": False,
            "exact_arithmetic": "fractions.Fraction",
            "candidate_search_rerun": False,
        },
    }


def main() -> int:
    payload = build_certificate_payload()
    if not ARTIFACT.is_file():
        raise FileNotFoundError("local-stability certificate is missing.")
    with ARTIFACT.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    if persisted != payload:
        raise RuntimeError(
            "local_stability_certificate.json is stale or inconsistent."
        )

    constants = payload["stability_constants"]
    radius_search = payload["radius_search"]
    print(f"selected_design: {payload['selected_design']}")
    print(f"exact_determinant: {payload['zero_jacobian']['exact_determinant']}")
    print(
        "exact_inverse_infinity_norm: "
        f"{payload['zero_jacobian']['exact_inverse_infinity_norm']}"
    )
    print(f"selected_radius: {radius_search['selected_radius']}")
    print(f"exact_delta: {radius_search['selected_delta']}")
    print(
        "exact_lower_lipschitz_constant: "
        f"{constants['exact_lower_lipschitz_constant']}"
    )
    print(
        "exact_inverse_lipschitz_bound: "
        f"{constants['exact_inverse_lipschitz_bound']}"
    )
    for assumption in payload["theorem_assumptions"]:
        print(f"assumption: {assumption}")
    print("common_scale_inverse_amplification: exact identities verified")
    print("local_stability_certificate: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

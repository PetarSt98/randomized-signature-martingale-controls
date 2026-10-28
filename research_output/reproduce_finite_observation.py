"""Reproduce the exact finite-observation certificate without writing files."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lie_reservoir import fraction_string  # noqa: E402
from lie_reservoir.certificates import _determinant  # noqa: E402
from lie_reservoir.finite_observation import (  # noqa: E402
    _prescribed_design_search,
    certify_exact_design,
    common_rescale_design,
    exact_matrix_rank,
    exact_truncated_observation_map,
    exact_zero_jacobian,
    exact_zero_jacobian_from_tensor,
    exact_joint_state_scale_matrix,
    fallback_candidate_state_scale_pairs,
    hall_basis_degrees,
    hall_basis_labels,
    noncertifying_spectral_diagnostics,
    primary_candidate_state_scale_pairs,
    select_exact_design,
    single_scale_field_jet_matrix,
)
from research_output.reproduce_graded_reconstruction import (  # noqa: E402
    _semantic_json_sha256,
)


F = Fraction
ARTIFACT = ROOT / "research_output" / "finite_observation_certificate.json"
SOURCE_CERTIFICATE = ROOT / "research_output" / "certificate.json"
GRADED_CERTIFICATE = (
    ROOT / "research_output" / "graded_reconstruction_certificate.json"
)

ZERO_COORDINATES = (
    (F(0), F(0)),
    (F(0),),
    (F(0), F(0)),
    (F(0), F(0), F(0)),
)


def _fraction_vector(values: tuple[Fraction, ...]) -> list[str]:
    return [fraction_string(value) for value in values]


def _fraction_matrix(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> list[list[str]]:
    return [_fraction_vector(row) for row in matrix]


def _design_payload(design) -> list[dict[str, str]]:
    return [
        {"state": state_label, "scale": fraction_string(scale)}
        for state_label, scale in design
    ]


def _diagnostic_decimal(value: float) -> str:
    return format(value, ".12g")


def _relation_residual(
    matrix: tuple[tuple[Fraction, ...], ...],
    relation: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (entry * coefficient for entry, coefficient in zip(row, relation)),
            start=F(0),
        )
        for row in matrix
    )


def build_certificate_payload() -> dict[str, Any]:
    """Recompute every exact and diagnostic certificate field."""
    design, search_domain, subset_count, certified_count = (
        _prescribed_design_search()
    )
    if design != select_exact_design():
        raise RuntimeError("the deterministic design search is inconsistent.")

    tensor_jacobian = exact_zero_jacobian_from_tensor(design)
    represented_jacobian = exact_joint_state_scale_matrix(design)
    if tensor_jacobian != represented_jacobian:
        raise RuntimeError("the two exact Jacobian constructions disagree.")
    matrix, determinant, inverse = certify_exact_design(design)
    if matrix != tensor_jacobian or determinant == 0:
        raise RuntimeError("the selected exact design was not certified.")

    singular_values, condition_number = noncertifying_spectral_diagnostics(
        matrix
    )
    removed_ranks = [
        exact_matrix_rank(
            tuple(row for index, row in enumerate(matrix) if index != removed)
        )
        for removed in range(8)
    ]
    if removed_ranks != [7] * 8:
        raise RuntimeError("a seven-observation row-rank guard failed.")

    common_scale_checks: list[dict[str, Any]] = []
    for multiplier in (F(-2), F(1, 2), F(3)):
        scaled_design = common_rescale_design(design, multiplier)
        scaled_determinant = _determinant(exact_zero_jacobian(scaled_design))
        expected = multiplier**22 * determinant
        if scaled_determinant != expected:
            raise RuntimeError("the exponent-22 determinant law failed.")
        common_scale_checks.append(
            {
                "multiplier": fraction_string(multiplier),
                "exact_determinant": fraction_string(scaled_determinant),
                "h_power_22_identity_verified": True,
            }
        )

    single_scale_matrix = single_scale_field_jet_matrix(F(1))
    relation = (F(0), F(0), F(4), F(3), F(0), F(0), F(0), F(2))
    taylor_relation_residual = _relation_residual(
        single_scale_matrix,
        relation,
    )
    single_scale_rank = exact_matrix_rank(single_scale_matrix)
    first_seven_minor = tuple(tuple(row[:7]) for row in single_scale_matrix)
    rank_minor_determinant = _determinant(first_seven_minor)

    # Independent represented-field guard in ascending powers of u=tanh(y).
    h2_polynomial = (F(-1, 2), F(0), F(1, 2), F(0), F(0))
    h31_polynomial = (F(1, 2), F(0), F(0), F(0), F(-1, 2))
    h43_polynomial = (F(1, 4), F(0), F(-1), F(0), F(3, 4))
    polynomial_relation_residual = tuple(
        F(4) * h2_polynomial[index]
        + F(3) * h31_polynomial[index]
        + F(2) * h43_polynomial[index]
        for index in range(5)
    )
    if polynomial_relation_residual != (F(0),) * 5:
        raise RuntimeError("the represented-field polynomial identity failed.")
    if taylor_relation_residual != (F(0),) * 7 or single_scale_rank != 7:
        raise RuntimeError("the exact cross-degree scope guard failed.")
    if rank_minor_determinant == 0:
        raise RuntimeError("the exact rank-seven lower bound failed.")

    zero_observation = exact_truncated_observation_map(
        ZERO_COORDINATES,
        design,
    )
    if any(rational_part != 0 for _, rational_part in zero_observation):
        raise RuntimeError("the zero-signature centered response is not zero.")

    primary_candidates = primary_candidate_state_scale_pairs()
    fallback_candidates = fallback_candidate_state_scale_pairs()
    return {
        "schema_version": 1,
        "artifact_kind": "exact_finite_observation_certificate",
        "status": "PROVED",
        "scope": {
            "architecture": "certified scalar tanh witness",
            "input_dimension": 2,
            "state_dimension": 1,
            "maximum_degree": 4,
            "hall_coordinate_dimension": 8,
            "scalar_observation_count": 8,
            "observation_object": (
                "B: degree-four truncated finite-scale response"
            ),
        },
        "conventions": {
            "scaled_response": "Y^{y,r}(X)=Y^y(rX)",
            "associative_word_order": "chronological",
            "operator_order": "Gamma(i1...im)=D_Vi1...D_Vim",
            "operator_action": "D_V f=f' V",
            "bracket": "[V,W]=DW V-DV W",
            "tensor_exponential": "truncated concatenation exponential",
            "factorial_normalization": (
                "homogeneous coefficient equals normalized scale jet"
            ),
            "literal_response_representation": (
                "(q_multiplier,rational_increment) denotes "
                "q_multiplier*log(3)/2+rational_increment"
            ),
        },
        "authoritative_sources": {
            "homogeneous_certificate": {
                "path": "research_output/certificate.json",
                "semantic_sha256": _semantic_json_sha256(SOURCE_CERTIFICATE),
            },
            "graded_reconstruction_certificate": {
                "path": "research_output/graded_reconstruction_certificate.json",
                "semantic_sha256": _semantic_json_sha256(GRADED_CERTIFICATE),
            },
            "matrix_constructor": (
                "lie_reservoir.finite_observation.exact_zero_jacobian"
            ),
        },
        "hall_basis": [
            {"label": label, "degree": degree}
            for label, degree in zip(
                hall_basis_labels(),
                hall_basis_degrees(),
                strict=True,
            )
        ],
        "candidate_search": {
            "candidate_order": "state-major, then listed scale order",
            "primary_states": ["-log(3)/2", "0", "log(3)/2"],
            "primary_scales": ["1", "2", "3", "4"],
            "primary_candidate_pairs": _design_payload(primary_candidates),
            "primary_candidate_count": len(primary_candidates),
            "eight_subset_count": subset_count,
            "exact_nonsingular_subset_count": certified_count,
            "selected_domain": search_domain,
            "fallback_used": search_domain == "fallback",
            "fallback_scales": ["-3", "-2", "-1", "1", "2", "3"],
            "fallback_candidate_count": len(fallback_candidates),
            "selection_rule": [
                "exact determinant must be nonzero",
                "fewer distinct absolute scale values",
                "lexicographically smaller distinct absolute scale set",
                "lower non-certifying binary64 2-norm condition number",
                "lexicographic ordered-pair list",
            ],
        },
        "selected_design": _design_payload(design),
        "zero_observation": [
            {
                "q_multiplier": q_multiplier,
                "rational_increment": fraction_string(rational_part),
            }
            for q_multiplier, rational_part in zero_observation
        ],
        "zero_jacobian": {
            "hall_column_order": list(hall_basis_labels()),
            "exact_matrix": _fraction_matrix(matrix),
            "tensor_and_lie_constructions_equal": True,
            "exact_determinant": fraction_string(determinant),
            "exact_determinant_nonzero": True,
            "exact_inverse": _fraction_matrix(inverse),
            "left_inverse_verified": True,
            "right_inverse_verified": True,
            "single_row_removal_exact_ranks": removed_ranks,
        },
        "noncertifying_diagnostics": {
            "arithmetic": "binary64",
            "purpose": "selection diagnostic only, not rank certification",
            "reported_significant_digits": 12,
            "singular_values": [
                _diagnostic_decimal(value) for value in singular_values
            ],
            "condition_number_2": _diagnostic_decimal(condition_number),
        },
        "common_scale_corollary": {
            "degree_multiplicities": {"1": 2, "2": 1, "3": 2, "4": 3},
            "degree_sum": 22,
            "law": "det B(h)=h^22 det B(1)",
            "exact_checks": common_scale_checks,
        },
        "cross_degree_scope_guard": {
            "relation_hall_coefficients": _fraction_vector(relation),
            "relation": (
                "rho(4[X1,X2]+3[X1,[X1,X2]]"
                "+2[[[X1,X2],X2],X2])=0"
            ),
            "represented_u_polynomials": {
                "variable": "u=tanh(y)",
                "coefficient_order": "ascending powers u^0,...,u^4",
                "[X1,X2]": _fraction_vector(h2_polynomial),
                "[X1,[X1,X2]]": _fraction_vector(h31_polynomial),
                "[[[X1,X2],X2],X2]": _fraction_vector(h43_polynomial),
            },
            "exact_polynomial_relation_residual": _fraction_vector(
                polynomial_relation_residual
            ),
            "represented_normalized_taylor_orders": list(range(7)),
            "exact_taylor_relation_residual": _fraction_vector(
                taylor_relation_residual
            ),
            "exact_represented_function_rank": single_scale_rank,
            "first_seven_column_minor_determinant": fraction_string(
                rank_minor_determinant
            ),
            "common_nonzero_scale_rank": 7,
            "conclusion": (
                "a full-rank design at the witness needs at least two scales"
            ),
        },
        "theorems": {
            "fixed_design_genericity": (
                "the fixed determinant is nontrivial real analytic in "
                "(a1,b1,a2,b2), hence nonzero almost everywhere"
            ),
            "local_inverse": (
                "the truncated polynomial has a local analytic inverse at zero"
            ),
            "dimension_lower_bound": (
                "no continuous locally injective U subset R^8 to R^M exists "
                "for M<8"
            ),
        },
        "exclusions": [
            "actual full CDE observations (object C)",
            "levels at least five",
            "full-CDE remainder estimates",
            "finite-difference reconstruction",
            "global invertibility",
            "universal measurement minimality",
        ],
        "reproduction": {
            "command": "python research_output/reproduce_finite_observation.py",
            "writes_files_by_default": False,
            "exact_arithmetic": "fractions.Fraction",
            "search": "deterministic exhaustive combinations",
        },
    }


def main() -> int:
    payload = build_certificate_payload()
    if not ARTIFACT.is_file():
        raise FileNotFoundError("finite observation certificate is missing.")
    with ARTIFACT.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    if persisted != payload:
        raise RuntimeError(
            "finite_observation_certificate.json is stale or inconsistent."
        )

    print(
        "candidate_search: "
        f"{payload['candidate_search']['exact_nonsingular_subset_count']}/"
        f"{payload['candidate_search']['eight_subset_count']} exact full rank"
    )
    print(f"selected_design: {payload['selected_design']}")
    print(
        "exact_determinant: "
        f"{payload['zero_jacobian']['exact_determinant']}"
    )
    print("jacobian_constructions: exact match")
    print("common_scale_law: h^22 verified")
    print("cross_degree_single_scale_rank: 7 verified")
    print("finite_observation_certificate: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

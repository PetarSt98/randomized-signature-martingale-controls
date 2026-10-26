"""Recompute and compare the finite exact fixed-zero Hall-rank certificate."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


# The default verification run must not create import caches either.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lie_reservoir.certificates import _determinant, fraction_string  # noqa: E402
from lie_reservoir.finite_observation import exact_matrix_rank  # noqa: E402
from lie_reservoir.free_lie import format_lie_node  # noqa: E402
from lie_reservoir.intrinsic_hall_rank import (  # noqa: E402
    MAX_DEGREE,
    MAX_INPUT_DIM,
    MAX_STATE_DIM,
    MAX_WORDS_AT_TOP_DEGREE,
    evaluate_hall_matrix,
    grading_witness,
    tanh_coefficients,
    witness_minor,
    zero_response_jacobian,
)


F = Fraction
ARTIFACT = ROOT / "research_output" / "intrinsic_hall_rank_certificate.json"
CASES = (
    (2, 2, None),
    (2, 3, None),
    (2, 4, None),
    (2, 5, None),
    (3, 3, None),
    (3, 4, None),
    (2, 4, 1),
    (2, 4, 4),
    (2, 4, 7),
    (2, 4, 9),
    (3, 3, 2),
    (3, 3, 8),
    (1, 5, 1),
    (1, 5, 2),
    (3, 1, 2),
    (3, 1, 3),
    (3, 1, 4),
)


def _vector(values) -> list[str]:
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in values):
        raise TypeError("certificate entries must be exact integers or Fractions.")
    return [fraction_string(F(value)) for value in values]


def _matrix(values) -> list[list[str]]:
    return [_vector(row) for row in values]


def _bias_entry(multiplier: Fraction) -> str:
    if multiplier == 0:
        return "0"
    if multiplier == 1:
        return "q"
    return f"({fraction_string(F(multiplier))})*q"


def _case_payload(input_dim: int, degree: int, state_dim: int | None) -> dict[str, Any]:
    witness = grading_witness(input_dim, degree, state_dim)
    matrix = evaluate_hall_matrix(witness)
    size = min(witness.state_dim, witness.nu)
    rows = tuple(range(size))
    columns = witness.retained_columns
    minor = witness_minor(witness, matrix)
    selected = tuple(tuple(matrix[row][column] for column in columns) for row in rows)
    if minor != selected:
        raise RuntimeError("the selected Hall minor differs from the matrix slice.")
    determinant = _determinant(minor)
    rank = exact_matrix_rank(matrix)
    if determinant != F(1, 2) ** size or rank != size:
        raise RuntimeError(f"exact rank or determinant failed for {(input_dim, degree, state_dim)}.")

    # These are independent expected structural guards, not a matrix constructor.
    for row, state_grade in enumerate(witness.state_degrees):
        for column, hall_grade in enumerate(witness.hall_degrees):
            if (state_grade == 0 or state_grade > hall_grade) and matrix[row][column] != 0:
                raise RuntimeError("an inactive row or below-diagonal grade block is nonzero.")
    for minor_row, state_row in enumerate(rows):
        for minor_column, hall_column in enumerate(columns):
            if witness.state_degrees[state_row] == witness.hall_degrees[hall_column]:
                expected = F(int(minor_row == minor_column), 2)
                if minor[minor_row][minor_column] != expected:
                    raise RuntimeError("a diagonal grade block differs from (1/2) I.")

    scale_checks = []
    if (input_dim, degree, witness.state_dim) == (2, 4, 8):
        for scale in (F(0), F(-2), F(1, 2), F(3)):
            jacobian = zero_response_jacobian(witness, scale)
            expected = tuple(
                tuple(entry * scale**hall_grade for entry, hall_grade in zip(row, witness.hall_degrees, strict=True))
                for row in matrix
            )
            if jacobian != expected:
                raise RuntimeError("the truncated-response zero-Jacobian scaling check failed.")
            scaled_determinant = _determinant(witness_minor(witness, jacobian))
            if scaled_determinant != determinant * scale ** sum(witness.hall_degrees):
                raise RuntimeError("the selected scaled Jacobian determinant is inconsistent.")
            scale_checks.append({
                "scale": fraction_string(scale),
                "exact_determinant": fraction_string(scaled_determinant),
                "column_scaling_verified": True,
            })

    relation = "equal"
    if witness.state_dim < witness.nu:
        relation = "graded_quotient"
    elif witness.state_dim > witness.nu:
        relation = "inactive_embedding"

    return {
        "input_dimension": input_dim,
        "maximum_degree": degree,
        "generator_jet_order": degree - 1,
        "state_dimension": witness.state_dim,
        "hall_dimension": witness.nu,
        "homogeneous_dimensions": list(witness.homogeneous_dimensions),
        "construction": relation,
        "hall_basis": [
            {"label": format_lie_node(node), "degree": hall_grade}
            for node, hall_grade in zip(witness.basis, witness.hall_degrees, strict=True)
        ],
        "state_degrees": list(witness.state_degrees),
        "active_grading_diagonal_with_zero_padding": _vector(witness.grading),
        "retained_hall_columns": list(columns),
        "projected_generators": [_vector(vector) for vector in witness.projected_generators],
        "generator_adjoint_matrices": [_matrix(matrix) for matrix in witness.adjoints],
        "generator_lambda_matrices": [_matrix(matrix) for matrix in witness.lambdas],
        "exact_parameters": {
            "q": "atanh(1/2)=log(3)/2",
            "weights": [_matrix(matrix) for matrix in witness.weights],
            "bias_q_multipliers": [_vector(vector) for vector in witness.bias_q_multipliers],
            "bias_entries": [
                [_bias_entry(multiplier) for multiplier in vector]
                for vector in witness.bias_q_multipliers
            ],
        },
        "initial_state": ["0"] * witness.state_dim,
        "exact_hall_evaluation_matrix": _matrix(matrix),
        "selected_rows": list(rows),
        "selected_columns": list(columns),
        "exact_selected_minor": _matrix(minor),
        "exact_determinant": fraction_string(determinant),
        "exact_rank": rank,
        "checks": {
            "selected_minor_matches_matrix_slice": True,
            "block_upper_triangular": True,
            "diagonal_blocks_equal_half_identity": True,
            "inactive_rows_zero": True,
            "determinant_equals_two_to_minus_minor_size": True,
            "rank_equals_minimum_dimension": True,
        },
        "truncated_response_zero_jacobian_scale_checks": scale_checks,
    }


def build_certificate_payload() -> dict[str, Any]:
    """Compute actual finite jets and Hall matrices, then check their ranks."""
    coefficients = [tanh_coefficients(value, 7) for value in (F(0), F(1, 2))]
    if coefficients[0] != (F(0), F(1), F(0), F(-1, 3), F(0), F(2, 15), F(0), F(-17, 315)):
        raise RuntimeError("the normalized tanh jet at zero failed its exact guard.")
    if coefficients[1][:2] != (F(1, 2), F(3, 4)):
        raise RuntimeError("the biased tanh value or first derivative is incorrect.")
    cases = []
    for input_dim, degree, state_dim in CASES:
        try:
            cases.append(_case_payload(input_dim, degree, state_dim))
        except Exception as error:
            dimension = "nu" if state_dim is None else str(state_dim)
            raise RuntimeError(
                f"intrinsic Hall-rank check failed for d={input_dim}, "
                f"L={degree}, D={dimension}: {error}"
            ) from error
    return {
        "schema_version": 1,
        "artifact_kind": "exact_finite_fixed_zero_hall_rank_witnesses",
        "status": "EXACT_FINITE_WITNESS_CHECKS",
        "scope": {
            "architecture": "V_i(x)=tanh(A_i x+b_i), componentwise, full biased parameter family",
            "initialization": "fixed zero",
            "matrix": "H_theta=[rho(H_1)(0) ... rho(H_nu)(0)]",
            "claim_checked": "rank H_theta=min(D,nu) at the listed finite exact witnesses",
            "general_theorem_source": "research_output/intrinsic_hall_rank_theorem.md",
            "genericity": "the all-d,D,L proof and analytic zero-set argument are in the theorem note; finite cases alone do not prove them",
        },
        "conventions": {
            "letter_indices": "zero-based internally; labels X1,...,Xd",
            "hall_order": "increasing degree, then lexicographic standard-Lyndon order",
            "matrix_indices": "zero-based rows and columns",
            "word_order": "chronological: earliest differential on the left",
            "tensor_product": "concatenation uv",
            "operator_order": "Gamma(i1...im)=D_Vi1...D_Vim; first letter outermost",
            "operator_action": "D_V f=Df V",
            "bracket": "[V,W]=DW V-DV W",
            "tanh_jet_coefficients": "ordinary Taylor coefficients f^(n)(0)/n!, including n=0",
            "scale_jet": "J_m=(1/m!) partial_r^m Y|_(r=0)",
            "empty_word": "known signature coefficient 1; Id(0)=0; no unital evaluation injectivity claimed",
        },
        "method": {
            "arithmetic": "fractions.Fraction; q remains symbolic",
            "witness": "A_i=-E^(-1) ad_(e_i) E, b_i=q e_i, q=atanh(1/2), on the graded quotient or active embedding",
            "matrix_construction": "actual finite tanh jets, chronological word differentiation and the existing Hall word expansions",
            "finite_jet_orders": "generator ordinary Taylor degree L-1; retain degree L-m after a word of length m",
            "rank_check": "exact elimination using existing determinant and matrix-rank utilities",
            "grading_padding": "E is inverted only on the positive-degree active space; inactive coordinates are recorded as zero",
        },
        "normalized_tanh_coefficients_through_order_seven": [
            {"tanh_value_at_expansion_point": fraction_string(value), "coefficients": _vector(jet)}
            for value, jet in zip((F(0), F(1, 2)), coefficients, strict=True)
        ],
        "implementation_limits": {
            "maximum_input_dimension": MAX_INPUT_DIM,
            "maximum_degree": MAX_DEGREE,
            "maximum_state_dimension": MAX_STATE_DIM,
            "maximum_words_at_top_degree": MAX_WORDS_AT_TOP_DEGREE,
            "meaning": "finite computational guardrails, not restrictions of the theorem",
        },
        "cases": cases,
        "limitations": [
            "the numerical values of q and tanh are never approximated in these checks",
            "the matrix need not equal (1/2) I; only its selected diagonal grade blocks do",
            "the nonlinear fields are not assumed to represent the truncated Lie or tensor quotient",
            "the local-recovery conclusion concerns the degree-L truncated polynomial, not the full CDE endpoint",
            "no linear-readout approximation, word-family independence at width nu, or conditioning bound is certified",
            "no conclusion for singular parameter laws follows from full-space analytic genericity alone",
        ],
        "reproduction": {
            "command": "python -B research_output/reproduce_intrinsic_hall_rank.py",
            "write_command": "python -B research_output/reproduce_intrinsic_hall_rank.py --write-artifacts",
            "writes_files_by_default": False,
            "write_target": "research_output/intrinsic_hall_rank_certificate.json",
            "comparison": "recompute and compare the entire deterministic JSON payload",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="write only the new intrinsic Hall-rank certificate after exact checks pass",
    )
    args = parser.parse_args()
    payload = build_certificate_payload()
    if args.write_artifacts:
        with ARTIFACT.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:
        if not ARTIFACT.is_file():
            raise FileNotFoundError("intrinsic Hall-rank certificate is missing; --write-artifacts creates it explicitly.")
        with ARTIFACT.open("r", encoding="utf-8") as handle:
            persisted = json.load(handle)
        if persisted != payload:
            raise RuntimeError("intrinsic_hall_rank_certificate.json is stale or inconsistent.")
    for case in payload["cases"]:
        print(
            f"d={case['input_dimension']} L={case['maximum_degree']} "
            f"D={case['state_dimension']} nu={case['hall_dimension']} "
            f"rank={case['exact_rank']} det_minor={case['exact_determinant']} "
            f"construction={case['construction']}"
        )
    print(f"intrinsic_hall_rank_certificate: {'written' if args.write_artifacts else 'verified'} ({len(payload['cases'])} exact finite cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the exact graded-reconstruction certificate without writing files."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lie_reservoir import (  # noqa: E402
    degree_exact_certificate,
    exact_tanh_witness_graded_jets,
    fraction_string,
    reconstruct_tanh_witness_hall_coordinates,
)
from lie_reservoir.graded_reconstruction import (  # noqa: E402
    _reconstruction_steps,
    exact_reconstruction_inverses,
    exact_reconstruction_matrices,
)


F = Fraction
ARTIFACT = ROOT / "research_output" / "graded_reconstruction_certificate.json"
SOURCE_CERTIFICATE = ROOT / "research_output" / "certificate.json"

GATE_COORDINATES = (
    (
        (F(1), F(-1)),
        (F(2),),
        (F(3), F(-2)),
        (F(-1), F(2), F(-3)),
    ),
    (
        (F(1, 2), F(-2, 3)),
        (F(5, 7),),
        (F(-3, 5), F(4, 9)),
        (F(7, 11), F(-8, 13), F(9, 14)),
    ),
    (
        (F(-5, 3), F(2, 7)),
        (F(-11, 4),),
        (F(13, 8), F(-17, 9)),
        (F(19, 10), F(-23, 12), F(29, 15)),
    ),
)


def _fraction_vector(values: tuple[Fraction, ...]) -> list[str]:
    return [fraction_string(value) for value in values]


def _fraction_blocks(
    blocks: tuple[tuple[Fraction, ...], ...],
) -> list[list[str]]:
    return [_fraction_vector(block) for block in blocks]


def _fraction_matrix(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> list[list[str]]:
    return [_fraction_vector(row) for row in matrix]


def _matrix_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (
                    left[row][index] * right[index][column]
                    for index in range(len(right))
                ),
                start=F(0),
            )
            for column in range(len(right))
        )
        for row in range(len(left))
    )


def _identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(F(int(row == column)) for column in range(size))
        for row in range(size)
    )


def _semantic_json_sha256(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _all_coordinates_nonzero(
    coordinates: tuple[tuple[Fraction, ...], ...],
) -> bool:
    return all(value != 0 for block in coordinates for value in block)


def _round_trip_payload(
    coordinates: tuple[tuple[Fraction, ...], ...],
    *,
    include_steps: bool,
) -> dict[str, Any]:
    jets = exact_tanh_witness_graded_jets(coordinates)
    recovered = reconstruct_tanh_witness_hall_coordinates(jets)
    if recovered != coordinates:
        raise RuntimeError("the exact graded reconstruction round trip failed.")

    payload: dict[str, Any] = {
        "input_hall_coordinates": _fraction_blocks(coordinates),
        "observed_normalized_jets": _fraction_blocks(jets),
        "recovered_hall_coordinates": _fraction_blocks(recovered),
        "exact_equality_verified": True,
    }
    if include_steps:
        payload["recursive_steps"] = [
            {
                "degree": step.degree,
                "basis_labels": list(step.basis_labels),
                "point_labels": list(step.point_labels),
                "observed_jet_vector": _fraction_vector(
                    step.observed_jet_vector
                ),
                "lower_level_correction": _fraction_vector(
                    step.lower_level_correction
                ),
                "residual": _fraction_vector(step.residual),
                "recovered_hall_coefficients": _fraction_vector(
                    step.recovered_hall_coefficients
                ),
            }
            for step in _reconstruction_steps(jets)
        ]
    return payload


def build_certificate_payload() -> dict[str, Any]:
    """Recompute every persisted exact datum from authoritative code."""
    matrices = exact_reconstruction_matrices()
    inverses = exact_reconstruction_inverses()
    all_gate_coordinates_nonzero = all(
        _all_coordinates_nonzero(coordinates)
        for coordinates in GATE_COORDINATES
    )
    if not all_gate_coordinates_nonzero:
        raise RuntimeError("every gate vector must have all eight coordinates nonzero.")

    degrees: list[dict[str, Any]] = []
    for degree, matrix, inverse in zip(
        range(1, 5),
        matrices,
        inverses,
        strict=True,
    ):
        certificate = degree_exact_certificate(degree)
        if matrix != certificate.exact_minor:
            raise RuntimeError("a reconstruction matrix is not authoritative.")
        identity = _identity(len(matrix))
        left_identity = _matrix_product(inverse, matrix) == identity
        right_identity = _matrix_product(matrix, inverse) == identity
        if not left_identity or not right_identity:
            raise RuntimeError("an exact reconstruction inverse failed.")
        degrees.append(
            {
                "degree": degree,
                "basis_labels": list(certificate.basis_labels),
                "point_labels": list(certificate.evaluation_points),
                "authoritative_matrix": _fraction_matrix(matrix),
                "recomputed_inverse": _fraction_matrix(inverse),
                "left_identity_verified": left_identity,
                "right_identity_verified": right_identity,
            }
        )

    round_trips = [
        _round_trip_payload(coordinates, include_steps=index == 0)
        for index, coordinates in enumerate(GATE_COORDINATES)
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "exact_graded_reconstruction_certificate",
        "status": "PROVED",
        "scope": {
            "architecture": "certified scalar tanh witness",
            "input_dimension": 2,
            "state_dimension": 1,
            "degrees": [1, 2, 3, 4],
            "total_hall_dimension": 8,
            "observation_object": "A: ideal normalized graded scale jets",
        },
        "conventions": {
            "associative_words": (
                "chronological zero-based tuples (i1,...,im)"
            ),
            "tensor_product": "noncommutative concatenation, truncated at 4",
            "operator_representation": (
                "Gamma(i1...im)=D_Vi1 ... D_Vim; Gamma is a homomorphism"
            ),
            "operator_action": "D_V f = f' V",
            "free_lie_bracket": "[X1,X2]=X1X2-X2X1",
            "vector_field_bracket": "[V,W]=DW V-DV W",
            "factorial_normalization": (
                "J_m=(1/m!) partial_r^m Y_T(rX)|_{r=0}"
            ),
            "log_signature": "S^{<=4}=exp_tensor(ell_1+...+ell_4)",
        },
        "authoritative_source": {
            "matrix_constructor": (
                "lie_reservoir.certificates.degree_exact_certificate"
            ),
            "certificate_path": "research_output/certificate.json",
            "certificate_semantic_sha256": _semantic_json_sha256(
                SOURCE_CERTIFICATE
            ),
        },
        "degree_data": degrees,
        "gate_round_trips": round_trips,
        "gate": {
            "all_eight_coordinates_nonzero_in_each_vector": (
                all_gate_coordinates_nonzero
            ),
            "nontrivial_rational_vector_count": len(round_trips),
            "exact_round_trips_verified": True,
        },
        "exclusions": [
            "degree-four truncated finite-scale responses (object B)",
            "actual full CDE responses and levels at least five (object C)",
            "finite-difference reconstruction",
            "local stability constants",
            "global inversion",
            "measurement or reservoir-run minimality",
        ],
        "reproduction_command": (
            "python research_output/reproduce_graded_reconstruction.py"
        ),
    }


def main() -> int:
    payload = build_certificate_payload()
    if not ARTIFACT.is_file():
        raise FileNotFoundError("graded reconstruction certificate is missing.")
    with ARTIFACT.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    if persisted != payload:
        raise RuntimeError(
            "graded_reconstruction_certificate.json is stale or inconsistent."
        )

    primary = payload["gate_round_trips"][0]
    print(f"input_coordinates: {primary['input_hall_coordinates']}")
    print(f"recovered_coordinates: {primary['recovered_hall_coordinates']}")
    print("authoritative_matrices: recomputed")
    print("exact_inverses: recomputed")
    print(
        "all_nonzero_gate_vectors: "
        f"{payload['gate']['nontrivial_rational_vector_count']} verified"
    )
    print("certificate_artifact: verified")
    print("exact_round_trip: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

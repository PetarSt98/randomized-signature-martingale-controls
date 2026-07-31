"""Reproduce the saved candidate and exact homogeneous certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import torch


def _find_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for folder in (start, *start.parents):
        if (folder / "lie_reservoir").is_dir():
            return folder
    raise RuntimeError("could not locate the lie_reservoir package.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lie_reservoir import (  # noqa: E402
    ShallowBiasedTanhConfig,
    ShallowBiasedTanhParameters,
    build_homogeneous_evaluation_matrix,
    classify_certificate,
    fraction_string,
    report_fraction_as_float,
    report_minor_as_float,
    through_degree_four_exact_certificate,
)


TARGET_DIR = REPO_ROOT / "research_output" / "selected_target"
CERTIFICATE_JSON = REPO_ROOT / "research_output" / "certificate.json"
CERTIFICATE_NOTE = REPO_ROOT / "research_output" / "certificate_note.md"

EXPECTED_BASIS = (
    "[X1,[X1,[X1,X2]]]",
    "[X1,[[X1,X2],X2]]",
    "[[[X1,X2],X2],X2]",
)
EXPECTED_SOURCE_WEIGHTS = (
    -1.1257398491596682,
    -0.005658299322056218,
)
EXPECTED_SOURCE_BIASES = (
    -1.3974639278267806,
    1.4364079096034785,
)
EXPECTED_SOURCE_SPECTRUM = (
    5.87794611591522,
    4.830316426888776,
    2.64088207597699,
)
EXPECTED_SOURCE_ROWS = (18, 32, 22)
EXPECTED_SOURCE_DETERMINANT = -5.041803997872895


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            payload,
            output_file,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


def _verify_source_candidate() -> dict[str, Any]:
    manifest_path = TARGET_DIR / "manifest.json"
    manifest = _load_json(manifest_path)
    for artifact in manifest["artifacts"].values():
        artifact_path = REPO_ROOT / artifact["path"]
        if _sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {artifact_path}")

    parameters = np.load(TARGET_DIR / "parameters.npz")
    evaluation_points = np.load(TARGET_DIR / "evaluation_points.npy")
    saved_matrix = np.load(TARGET_DIR / "evaluation_matrix.npy")
    saved_spectrum = np.load(TARGET_DIR / "singular_values.npy")
    saved_minor = np.load(TARGET_DIR / "candidate_minor.npz")
    minor_metadata = _load_json(TARGET_DIR / "candidate_minor.json")
    basis_labels = tuple(_load_json(TARGET_DIR / "basis_labels.json"))

    expected_manifest = {
        "architecture": "shallow_biased_tanh",
        "input_dim": 2,
        "state_dim": 1,
        "target_degree": 4,
        "witt_dimension": 3,
        "weight_seed": 1729,
        "point_seed": 2719,
        "extension_point_seed": 1000002719,
        "point_count": 42,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise RuntimeError("saved manifest differs from the selected target.")
    if basis_labels != EXPECTED_BASIS:
        raise RuntimeError("saved basis ordering differs from the selected target.")
    if tuple(parameters["weights"].reshape(-1)) != EXPECTED_SOURCE_WEIGHTS:
        raise RuntimeError("saved weights differ from the selected target.")
    if tuple(parameters["biases"].reshape(-1)) != EXPECTED_SOURCE_BIASES:
        raise RuntimeError("saved biases differ from the selected target.")
    if tuple(saved_spectrum) != EXPECTED_SOURCE_SPECTRUM:
        raise RuntimeError("saved spectrum differs from the selected target.")
    if tuple(saved_minor["selected_scalar_rows"]) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("saved row selection differs from the selected target.")
    if tuple(minor_metadata["selected_scalar_rows"]) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("minor metadata row selection is inconsistent.")

    _, recomputed_spectrum, pivots = scipy.linalg.qr(
        saved_matrix.T,
        mode="economic",
        pivoting=True,
    )
    del recomputed_spectrum
    qr_rows = tuple(int(row) for row in pivots[: saved_matrix.shape[1]])
    if qr_rows != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("pivoted QR did not reproduce the saved rows.")
    selected_columns = tuple(range(saved_matrix.shape[1]))
    reconstructed_minor = saved_matrix[np.asarray(qr_rows), :]
    if not np.array_equal(reconstructed_minor, saved_minor["matrix"]):
        raise RuntimeError("saved minor is not the deterministic matrix slice.")
    floating_determinant = float(np.linalg.det(reconstructed_minor))
    if not math.isclose(
        floating_determinant,
        EXPECTED_SOURCE_DETERMINANT,
        rel_tol=2e-15,
        abs_tol=0.0,
    ):
        raise RuntimeError("saved floating determinant was not reproduced.")
    if round(float(minor_metadata["condition_number"]), 3) != 1.986:
        raise RuntimeError("saved condition number differs from the summary.")
    singular_values = np.linalg.svd(saved_matrix, compute_uv=False)
    if not np.allclose(singular_values, saved_spectrum, rtol=2e-15, atol=0.0):
        raise RuntimeError("saved full singular spectrum was not reproduced.")

    config = ShallowBiasedTanhConfig(input_dim=2, state_dim=1)
    torch_parameters = ShallowBiasedTanhParameters(
        weights=torch.tensor(parameters["weights"], dtype=torch.float64),
        biases=torch.tensor(parameters["biases"], dtype=torch.float64),
    )
    torch_points = torch.tensor(evaluation_points, dtype=torch.float64)
    rebuilt_matrix = build_homogeneous_evaluation_matrix(
        config=config,
        parameters=torch_parameters,
        degree=4,
        evaluation_points=torch_points,
    )
    rebuilt_values = rebuilt_matrix.detach().cpu().numpy()
    if not np.allclose(rebuilt_values, saved_matrix, rtol=2e-14, atol=2e-14):
        raise RuntimeError("autodiff did not reproduce the saved matrix.")

    return {
        "status": "numerical_candidate_only",
        "manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "architecture": manifest["architecture"],
        "input_dim": manifest["input_dim"],
        "state_dim": manifest["state_dim"],
        "degree": manifest["target_degree"],
        "weight_seed": manifest["weight_seed"],
        "point_seed": manifest["point_seed"],
        "extension_point_seed": manifest["extension_point_seed"],
        "point_count": manifest["point_count"],
        "basis_labels": list(basis_labels),
        "selected_rows": list(qr_rows),
        "selected_columns": list(selected_columns),
        "floating_determinant": EXPECTED_SOURCE_DETERMINANT,
        "floating_condition_number": minor_metadata["condition_number"],
        "verification": {
            "artifact_hashes_match": True,
            "parameters_match": True,
            "basis_order_matches": True,
            "pivoted_qr_rows_reproduced": True,
            "minor_slice_matches": True,
            "spectrum_matches": True,
            "autodiff_matrix_matches": True,
        },
        "warning": "This source witness remains numerical_candidate_only.",
    }


def _fraction_payload(value: Any) -> dict[str, Any]:
    return {
        "exact_fraction": fraction_string(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
        "nonzero": value != 0,
    }


def _minor_conversion_payload(report: Any) -> dict[str, Any]:
    return {
        "values": [list(row) for row in report.values],
        "exact_values": [list(row) for row in report.exact_values],
        "conversion": report.conversion,
        "purpose": report.purpose,
        "warning": report.warning,
    }


def _value_conversion_payload(report: Any) -> dict[str, Any]:
    return {
        "value": report.value,
        "exact_value": report.exact_value,
        "conversion": report.conversion,
        "purpose": report.purpose,
        "warning": report.warning,
    }


def _point_value(point: str, q_value: float) -> float:
    values = {
        "-log(3)/2": -q_value,
        "0": 0.0,
        "log(3)/2": q_value,
    }
    try:
        return values[point]
    except KeyError as exc:
        raise ValueError(f"unsupported exact evaluation point: {point}") from exc


def _exact_numerical_cross_checks(combined: Any) -> dict[str, Any]:
    q_value = math.log(3.0) / 2.0
    q_conversion = {
        "value": q_value,
        "exact_value": "log(3)/2",
        "conversion": "exact expression evaluated with math.log in binary64",
        "purpose": "construct the numerical autodiff cross-checks only",
        "warning": "This conversion is not used for certification.",
    }
    config = ShallowBiasedTanhConfig(input_dim=2, state_dim=1)
    parameters = ShallowBiasedTanhParameters(
        weights=torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64),
        biases=torch.tensor([[0.0], [q_value]], dtype=torch.float64),
    )
    homogeneous_checks = []
    numerical_product = 1.0
    for certificate in combined.homogeneous_certificates:
        points = torch.tensor(
            [
                [_point_value(point, q_value)]
                for point in certificate.evaluation_points
            ],
            dtype=torch.float64,
        )
        numerical_minor = build_homogeneous_evaluation_matrix(
            config=config,
            parameters=parameters,
            degree=certificate.degree,
            evaluation_points=points,
        ).detach().cpu().numpy()
        minor_conversion = report_minor_as_float(
            certificate.exact_minor,
            purpose=(
                f"compare the exact degree-{certificate.degree} minor with "
                "the PyTorch autodiff minor"
            ),
        )
        determinant_conversion = report_fraction_as_float(
            certificate.exact_determinant,
            purpose=(
                f"compare the exact degree-{certificate.degree} determinant "
                "with a numerical determinant"
            ),
        )
        converted_minor = np.asarray(
            minor_conversion.values,
            dtype=np.float64,
        )
        if not np.allclose(
            numerical_minor,
            converted_minor,
            rtol=2e-14,
            atol=2e-14,
        ):
            raise RuntimeError(
                f"degree-{certificate.degree} exact and numerical minors "
                "disagree."
            )
        numerical_determinant = float(np.linalg.det(numerical_minor))
        if not math.isclose(
            numerical_determinant,
            determinant_conversion.value,
            rel_tol=3e-14,
            abs_tol=3e-14,
        ):
            raise RuntimeError(
                f"degree-{certificate.degree} exact and numerical "
                "determinants disagree."
            )
        numerical_rank = int(
            np.linalg.matrix_rank(numerical_minor, tol=1e-12)
        )
        if numerical_rank != len(certificate.basis_labels):
            raise RuntimeError(
                f"degree-{certificate.degree} witness is not numerically "
                "full rank."
            )
        numerical_only_status = classify_certificate(
            exact_determinant=None,
            numerical_determinant=numerical_determinant,
        )
        if numerical_only_status != "numerical_candidate_only":
            raise RuntimeError("numerical evidence was silently upgraded.")
        singular_values = np.linalg.svd(
            numerical_minor,
            compute_uv=False,
        )
        condition_number = float(
            singular_values[0] / singular_values[-1]
        )
        numerical_product *= numerical_determinant
        homogeneous_checks.append(
            {
                "degree": certificate.degree,
                "minor_conversion": _minor_conversion_payload(
                    minor_conversion
                ),
                "determinant_conversion": _value_conversion_payload(
                    determinant_conversion
                ),
                "numerical_rank": numerical_rank,
                "condition_number_rounded": float(
                    f"{condition_number:.8g}"
                ),
                "numerical_evidence_status_in_isolation": (
                    numerical_only_status
                ),
            }
        )

    product_conversion = report_fraction_as_float(
        combined.exact_determinant_product,
        purpose="compare the exact determinant product with its numerical check",
    )
    if not math.isclose(
        numerical_product,
        product_conversion.value,
        rel_tol=1e-13,
        abs_tol=1e-15,
    ):
        raise RuntimeError("exact and numerical determinant products disagree.")
    return {
        "q_conversion": q_conversion,
        "homogeneous": homogeneous_checks,
        "combined_product_conversion": _value_conversion_payload(
            product_conversion
        ),
        "numerical_product": numerical_product,
        "reproducer_requires_autodiff_agreement": True,
        "comparison_absolute_tolerance": 2e-14,
        "warning": (
            "All floating values in this section are non-certifying "
            "cross-checks only."
        ),
    }


def _homogeneous_certificate_payload(certificate: Any) -> dict[str, Any]:
    return {
        "degree": certificate.degree,
        "basis_labels": list(certificate.basis_labels),
        "evaluation_points": list(certificate.evaluation_points),
        "selected_rows": list(certificate.selected_rows),
        "selected_columns": list(certificate.selected_columns),
        "exact_matrix": [
            [fraction_string(value) for value in row]
            for row in certificate.exact_minor
        ],
        "determinant_representation": _fraction_payload(
            certificate.exact_determinant
        ),
        "certification_method": certificate.certification_method,
        "status": certificate.status,
    }


def build_certificate_payload() -> dict[str, Any]:
    """Verify both witnesses and return the deterministic certificate record."""
    source_candidate = _verify_source_candidate()
    combined = through_degree_four_exact_certificate()
    certificates = combined.homogeneous_certificates
    numerical_cross_checks = _exact_numerical_cross_checks(combined)
    first_certificate = certificates[0]
    fixed_points = {
        str(certificate.degree): list(certificate.evaluation_points)
        for certificate in certificates
    }
    return {
        "schema_version": 2,
        "architecture": combined.architecture,
        "input_dim": combined.input_dim,
        "state_dim": combined.state_dim,
        "through_degree": 4,
        "certified_homogeneous_degrees": list(combined.degrees),
        "status_scope": "all_homogeneous_degrees_1_through_4",
        "bracket_convention": "[V,W](x) = DW(x)V(x) - DV(x)W(x)",
        "constant_definition": first_certificate.constant_definition,
        "activation_identities": list(
            first_certificate.activation_identities
        ),
        "exact_parameters": {
            "weights": {
                "shape": [2, 1, 1],
                "entries": list(first_certificate.weight_entries),
            },
            "biases": {
                "shape": [2, 1],
                "entries": list(first_certificate.bias_entries),
            },
        },
        "homogeneous_certificates": [
            _homogeneous_certificate_payload(certificate)
            for certificate in certificates
        ],
        "combined_determinant_product": _fraction_payload(
            combined.exact_determinant_product
        ),
        "symbolic_reconstruction": (
            "exact Fraction Taylor jets generated from y'=a*(1-y^2), "
            "combined recursively with [V,W]=W'V-V'W"
        ),
        "certification_method": combined.certification_method,
        "status": combined.status,
        "conclusion": combined.conclusion,
        "finite_evaluation_lemma": (
            "If evaluations of vector fields F_1,...,F_r at finitely many "
            "points form a full-column-rank matrix, then F_1,...,F_r are "
            "linearly independent as functions."
        ),
        "reproduction_command": (
            "python research_output/reproduce_certificate.py"
        ),
        "source_numerical_candidate": source_candidate,
        "reported_numerical_cross_checks": numerical_cross_checks,
        "analytic_implication": {
            "fixed_evaluation_points_by_degree": fixed_points,
            "parameter_domain": (
                "(a1,b1,a2,b2) in the connected open set R^4"
            ),
            "analyticity": (
                "each selected homogeneous minor determinant D_m is real "
                "analytic because tanh of affine forms is real analytic on "
                "R, and state differentiation, Lie brackets, fixed-point "
                "evaluation, and determinants preserve real analyticity"
            ),
            "product_definition": "P = D_1 * D_2 * D_3 * D_4",
            "nonidentity_witness": (
                "P(1,0,0,log(3)/2) = -6075/16777216 != 0"
            ),
            "zero_set_conclusion": (
                "the set where at least one D_m vanishes is the zero set of "
                "P and has four-dimensional Lebesgue measure zero"
            ),
            "probability_conclusion": (
                "all four homogeneous components are injective "
                "simultaneously almost surely under any parameter law "
                "absolutely continuous with respect to four-dimensional "
                "Lebesgue measure"
            ),
            "excluded_laws": (
                "singular or fixed-bias laws and the literal finite-support "
                "law of a finite-precision seeded PRNG"
            ),
        },
        "scope_limit": (
            "This proves injectivity separately on L_{2,m} for m=1,2,3,4. "
            "It does not prove ungraded cross-degree independence or "
            "injectivity on their ungraded direct sum."
        ),
    }


def _homogeneous_note_section(certificate: dict[str, Any]) -> str:
    basis_lines = "\n".join(
        f"{index}. `{label}`"
        for index, label in enumerate(certificate["basis_labels"], start=1)
    )
    matrix_lines = "\n".join(
        "[" + "  ".join(row) + "]"
        for row in certificate["exact_matrix"]
    )
    points = ", ".join(certificate["evaluation_points"])
    determinant = certificate["determinant_representation"]["exact_fraction"]
    degree = certificate["degree"]
    return f"""### Degree {degree}

Evaluation points: `({points})`.

Basis:

{basis_lines}

```text
{matrix_lines}
```

Exact determinant: `{determinant} != 0`.
"""


def _certificate_note(payload: dict[str, Any]) -> str:
    sections = "\n".join(
        _homogeneous_note_section(certificate)
        for certificate in payload["homogeneous_certificates"]
    )
    product = payload["combined_determinant_product"]["exact_fraction"]
    return f"""# Exact homogeneous degree-one-to-four certificates

## Exact witness

Status: `exact_nonzero` for every homogeneous degree `m=1,2,3,4`.

For `shallow_biased_tanh`, `input_dim=2`, and `state_dim=1`, set

```text
q = atanh(1/2) = log(3)/2
(a1, b1, a2, b2) = (1, 0, 0, q)
V1(x) = tanh(x)
V2(x) = tanh(q) = 1/2
```

Exactly, `exp(2q)=3`, so `tanh(q)=1/2` and `tanh(-q)=-1/2`.
The repository's standard-Lyndon order is checked before each exact matrix is
constructed. Using `[V,W]=W'V-V'W`, the exact `Fraction` activation-jet
recursion gives:

{sections}
## Combined certificate

The exact determinant product is

```text
D1 * D2 * D3 * D4 = {product} != 0.
```

Therefore the representation is injective on each homogeneous free-Lie
component `L_{{2,m}}`, `m=1,2,3,4`.

Lemma: If evaluations of vector fields `F1,...,Fr` at finitely many points
form a full-column-rank matrix, then `F1,...,Fr` are linearly independent as
functions. Indeed, a functional linear relation evaluates to a vector in the
matrix nullspace, which is trivial at full column rank. Applying this to the
images of each complete Lyndon basis proves the stated homogeneous
injectivity.

## Analytic genericity argument

For the fixed degree-specific point sets above, each selected determinant
`D_m(a1,b1,a2,b2)` is real analytic on connected `R^4`. Their product
`P=D1*D2*D3*D4` is real analytic and the exact witness gives
`P(1,0,0,q)={product} != 0`, so `P` is not identically zero. Its zero set has
four-dimensional Lebesgue measure zero. Thus all four homogeneous components
are injective simultaneously almost surely under any parameter law absolutely
continuous with respect to four-dimensional Lebesgue measure.

This probability statement excludes singular or fixed-bias laws and the
literal finite-support distribution of a finite-precision seeded PRNG.

## Numerical checks and preserved atlas target

The reproduction program cross-checks every exact matrix through the PyTorch
autodiff pipeline. All exact-to-binary64 conversions are explicitly reported;
these checks are non-certifying numerical evidence only.

The original seed-1729 `42 x 3` degree-four atlas target, including pivoted-QR
rows `[18,32,22]`, is preserved separately as `numerical_candidate_only`.

## Scope limitation

This proves simultaneous degree-by-degree homogeneous injectivity for degrees
one through four. It does not prove ungraded cross-degree independence or
injectivity on the ungraded direct sum, because images from different degrees
could still cancel.

Reproduce with:

```text
python research_output/reproduce_certificate.py
```
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce exact homogeneous certificates through degree four."
        ),
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write certificate.json and certificate_note.md after verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_certificate_payload()
    note = _certificate_note(payload)
    if args.write_artifacts:
        _write_json(CERTIFICATE_JSON, payload)
        CERTIFICATE_NOTE.write_text(note, encoding="utf-8", newline="\n")
    else:
        if not CERTIFICATE_JSON.is_file() or not CERTIFICATE_NOTE.is_file():
            raise FileNotFoundError(
                "certificate artifacts are missing; rerun with --write-artifacts."
            )
        if _load_json(CERTIFICATE_JSON) != payload:
            raise RuntimeError("certificate.json is stale or inconsistent.")
        if CERTIFICATE_NOTE.read_text(encoding="utf-8") != note:
            raise RuntimeError("certificate_note.md is stale or inconsistent.")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method": payload["certification_method"],
                "certified_homogeneous_degrees": payload[
                    "certified_homogeneous_degrees"
                ],
                "exact_determinants": {
                    str(certificate["degree"]): certificate[
                        "determinant_representation"
                    ]["exact_fraction"]
                    for certificate in payload["homogeneous_certificates"]
                },
                "exact_determinant_product": payload[
                    "combined_determinant_product"
                ]["exact_fraction"],
                "source_rows": payload["source_numerical_candidate"][
                    "selected_rows"
                ],
                "source_status": payload["source_numerical_candidate"][
                    "status"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

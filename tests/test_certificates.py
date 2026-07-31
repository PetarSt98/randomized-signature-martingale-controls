"""Exact and numerical checks for homogeneous certificates through degree four."""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lie_reservoir import (  # noqa: E402
    ALLOWED_CERTIFICATE_STATUSES,
    ShallowBiasedTanhConfig,
    ShallowBiasedTanhParameters,
    build_homogeneous_evaluation_matrix,
    classify_certificate,
    degree_exact_certificate,
    degree_four_exact_certificate,
    report_fraction_as_float,
    report_minor_as_float,
    through_degree_four_exact_certificate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "research_output" / "selected_target"

EXPECTED_LABELS = {
    1: ("X1", "X2"),
    2: ("[X1,X2]",),
    3: ("[X1,[X1,X2]]", "[[X1,X2],X2]"),
    4: (
        "[X1,[X1,[X1,X2]]]",
        "[X1,[[X1,X2],X2]]",
        "[[[X1,X2],X2],X2]",
    ),
}

EXPECTED_POINTS = {
    1: ("0", "log(3)/2"),
    2: ("0",),
    3: ("-log(3)/2", "log(3)/2"),
    4: ("-log(3)/2", "0", "log(3)/2"),
}

EXPECTED_MATRICES = {
    1: (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
    ),
    2: ((Fraction(-1, 2),),),
    3: (
        (Fraction(15, 32), Fraction(3, 16)),
        (Fraction(15, 32), Fraction(-3, 16)),
    ),
    4: (
        (Fraction(-57, 128), Fraction(-3, 32), Fraction(3, 64)),
        (Fraction(-1, 2), Fraction(0), Fraction(1, 4)),
        (Fraction(-57, 128), Fraction(3, 32), Fraction(3, 64)),
    ),
}

EXPECTED_DETERMINANTS = {
    1: Fraction(-1, 4),
    2: Fraction(-1, 2),
    3: Fraction(-45, 256),
    4: Fraction(135, 8192),
}


def _assert_raises(
    exception_type: type[Exception],
    function: Callable[[], object],
) -> None:
    try:
        function()
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"expected {exception_type.__name__}, got {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"expected {exception_type.__name__}")


def _point_value(point: str, q_value: float) -> float:
    return {
        "-log(3)/2": -q_value,
        "0": 0.0,
        "log(3)/2": q_value,
    }[point]


def test_exact_homogeneous_matrices_determinants_and_product() -> None:
    combined = through_degree_four_exact_certificate()
    assert combined.degrees == (1, 2, 3, 4)
    assert combined.status == "exact_nonzero"
    assert combined.certification_method == (
        "exact_symbolic_determinant_simplification"
    )
    assert combined.exact_determinant_product == Fraction(
        -6075,
        16777216,
    )
    assert combined.conclusion == (
        "the representation is injective on each homogeneous free-Lie "
        "component L_{2,m}, m=1,2,3,4"
    )
    for certificate in combined.homogeneous_certificates:
        degree = certificate.degree
        assert certificate.basis_labels == EXPECTED_LABELS[degree]
        assert certificate.evaluation_points == EXPECTED_POINTS[degree]
        assert certificate.exact_minor == EXPECTED_MATRICES[degree]
        assert certificate.exact_determinant == EXPECTED_DETERMINANTS[degree]
        assert certificate.status == "exact_nonzero"
        assert certificate.selected_rows == tuple(
            range(len(EXPECTED_MATRICES[degree]))
        )
        assert certificate.selected_columns == tuple(
            range(len(EXPECTED_LABELS[degree]))
        )
        assert degree_exact_certificate(degree) == certificate
    assert combined.homogeneous_certificates[-1] == (
        degree_four_exact_certificate()
    )
    _assert_raises(ValueError, lambda: degree_exact_certificate(5))
    _assert_raises(TypeError, lambda: degree_exact_certificate(True))


def test_all_exact_and_autodiff_minors_agree_numerically() -> None:
    combined = through_degree_four_exact_certificate()
    q_value = math.log(3.0) / 2.0
    config = ShallowBiasedTanhConfig(input_dim=2, state_dim=1)
    parameters = ShallowBiasedTanhParameters(
        weights=torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64),
        biases=torch.tensor([[0.0], [q_value]], dtype=torch.float64),
    )
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
        )
        conversion = report_minor_as_float(
            certificate.exact_minor,
            purpose=(
                f"unit-test degree-{certificate.degree} autodiff comparison"
            ),
        )
        expected = torch.tensor(conversion.values, dtype=torch.float64)
        assert torch.allclose(
            numerical_minor,
            expected,
            rtol=2e-14,
            atol=2e-14,
        )
        determinant_conversion = report_fraction_as_float(
            certificate.exact_determinant,
            purpose=(
                f"unit-test degree-{certificate.degree} determinant comparison"
            ),
        )
        assert math.isclose(
            float(torch.linalg.det(numerical_minor).item()),
            determinant_conversion.value,
            rel_tol=3e-14,
            abs_tol=3e-14,
        )
        assert int(torch.linalg.matrix_rank(numerical_minor).item()) == len(
            certificate.basis_labels
        )


def test_source_and_exact_row_column_selections_are_reproducible() -> None:
    source_matrix = np.load(TARGET_DIR / "evaluation_matrix.npy")
    source_minor = np.load(TARGET_DIR / "candidate_minor.npz")
    _, _, pivots = scipy.linalg.qr(
        source_matrix.T,
        mode="economic",
        pivoting=True,
    )
    selected_rows = tuple(int(row) for row in pivots[: source_matrix.shape[1]])
    selected_columns = tuple(range(source_matrix.shape[1]))
    assert selected_rows == (18, 32, 22)
    assert selected_columns == (0, 1, 2)
    assert np.array_equal(
        source_matrix[np.asarray(selected_rows), :],
        source_minor["matrix"],
    )

    combined = through_degree_four_exact_certificate()
    for certificate in combined.homogeneous_certificates:
        assert certificate.selected_rows == tuple(
            range(len(certificate.exact_minor))
        )
        assert certificate.selected_columns == tuple(
            range(len(certificate.basis_labels))
        )


def test_status_is_not_upgraded_by_floating_point_evidence() -> None:
    assert set(ALLOWED_CERTIFICATE_STATUSES) == {
        "exact_nonzero",
        "taylor_coefficient_nonzero",
        "interval_certified_nonzero",
        "numerical_candidate_only",
        "inconclusive",
    }
    assert classify_certificate(
        exact_determinant=None,
        numerical_determinant=-5.041803997872895,
    ) == "numerical_candidate_only"
    assert classify_certificate(
        exact_determinant=None,
        numerical_determinant=None,
    ) == "inconclusive"
    assert classify_certificate(
        exact_determinant=Fraction(0),
        numerical_determinant=1.0,
    ) == "inconclusive"
    _assert_raises(
        TypeError,
        lambda: classify_certificate(
            exact_determinant=0.5,
            numerical_determinant=None,
        ),
    )


def test_every_exact_to_float_conversion_is_explicitly_reported() -> None:
    combined = through_degree_four_exact_certificate()
    for certificate in combined.homogeneous_certificates:
        assert all(
            isinstance(value, Fraction)
            for row in certificate.exact_minor
            for value in row
        )
        matrix_report = report_minor_as_float(
            certificate.exact_minor,
            purpose=(
                f"explicit degree-{certificate.degree} binary64 comparison"
            ),
        )
        determinant_report = report_fraction_as_float(
            certificate.exact_determinant,
            purpose=(
                f"explicit degree-{certificate.degree} determinant comparison"
            ),
        )
        assert matrix_report.purpose.startswith("explicit degree-")
        assert determinant_report.exact_value == str(
            certificate.exact_determinant
        )
        assert "not certification" in matrix_report.warning
        assert "not certification" in determinant_report.warning
    product_report = report_fraction_as_float(
        combined.exact_determinant_product,
        purpose="explicit combined-product comparison",
    )
    assert product_report.exact_value == "-6075/16777216"
    assert "not certification" in product_report.warning
    _assert_raises(
        ValueError,
        lambda: report_minor_as_float(
            combined.homogeneous_certificates[0].exact_minor,
            purpose="",
        ),
    )


def test_persisted_certificate_status_scope_and_checks_are_exact() -> None:
    certificate_path = REPO_ROOT / "research_output" / "certificate.json"
    with certificate_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    assert payload["schema_version"] == 2
    assert payload["status"] == "exact_nonzero"
    assert payload["status_scope"] == (
        "all_homogeneous_degrees_1_through_4"
    )
    assert payload["certified_homogeneous_degrees"] == [1, 2, 3, 4]
    assert payload["combined_determinant_product"]["exact_fraction"] == (
        "-6075/16777216"
    )
    persisted = {
        certificate["degree"]: certificate
        for certificate in payload["homogeneous_certificates"]
    }
    assert tuple(persisted) == (1, 2, 3, 4)
    for degree in (1, 2, 3, 4):
        assert persisted[degree]["basis_labels"] == list(
            EXPECTED_LABELS[degree]
        )
        assert persisted[degree]["evaluation_points"] == list(
            EXPECTED_POINTS[degree]
        )
        assert persisted[degree]["determinant_representation"][
            "exact_fraction"
        ] == str(EXPECTED_DETERMINANTS[degree])
    assert persisted[4]["exact_matrix"] == [
        [str(value) for value in row]
        for row in EXPECTED_MATRICES[4]
    ]
    assert payload["source_numerical_candidate"]["status"] == (
        "numerical_candidate_only"
    )
    checks = payload["reported_numerical_cross_checks"]
    assert checks["reproducer_requires_autodiff_agreement"] is True
    assert [check["degree"] for check in checks["homogeneous"]] == [
        1,
        2,
        3,
        4,
    ]
    assert all(
        check["numerical_evidence_status_in_isolation"]
        == "numerical_candidate_only"
        for check in checks["homogeneous"]
    )
    assert checks["combined_product_conversion"]["exact_value"] == (
        "-6075/16777216"
    )
    q_conversion = checks["q_conversion"]
    assert q_conversion["exact_value"] == "log(3)/2"
    assert q_conversion["conversion"] == (
        "exact expression evaluated with math.log in binary64"
    )
    assert q_conversion["purpose"] == (
        "construct the numerical autodiff cross-checks only"
    )
    assert "not used for certification" in q_conversion["warning"]


def test_persisted_theorem_and_scope_are_precise() -> None:
    certificate_path = REPO_ROOT / "research_output" / "certificate.json"
    with certificate_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    assert payload["conclusion"] == (
        "the representation is injective on each homogeneous free-Lie "
        "component L_{2,m}, m=1,2,3,4"
    )
    assert payload["finite_evaluation_lemma"] == (
        "If evaluations of vector fields F_1,...,F_r at finitely many "
        "points form a full-column-rank matrix, then F_1,...,F_r are "
        "linearly independent as functions."
    )
    implication = payload["analytic_implication"]
    assert implication["product_definition"] == "P = D_1 * D_2 * D_3 * D_4"
    assert implication["nonidentity_witness"] == (
        "P(1,0,0,log(3)/2) = -6075/16777216 != 0"
    )
    assert "Lebesgue measure zero" in implication["zero_set_conclusion"]
    assert "simultaneously almost surely" in (
        implication["probability_conclusion"]
    )
    assert "ungraded cross-degree independence" in payload["scope_limit"]


def _run_all() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()

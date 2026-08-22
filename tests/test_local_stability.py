from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from lie_reservoir.finite_observation import (
    certify_exact_design,
    common_rescale_design,
    exact_centered_observation_polynomials,
    exact_centered_truncated_observation_map,
    exact_common_scale_inverse,
    exact_common_scale_inverse_norm,
    exact_inverse_row_sums,
    exact_jacobian_difference_polynomials,
    exact_jacobian_row_degree_sums,
    exact_jacobian_variation_bound,
    exact_local_stability_constants,
    exact_matrix_infinity_norm,
    exact_stability_radius_table,
    hall_basis_degrees,
    stability_test_radii,
)
from research_output.reproduce_local_stability import (
    _validated_finite_certificate,
)


F = Fraction
ROOT = Path(__file__).resolve().parents[1]

DESIGN = (
    ("-log(3)/2", F(1)),
    ("-log(3)/2", F(2)),
    ("-log(3)/2", F(3)),
    ("0", F(2)),
    ("0", F(3)),
    ("log(3)/2", F(1)),
    ("log(3)/2", F(2)),
    ("log(3)/2", F(3)),
)

EXPECTED_INVERSE = (
    (
        F(-18, 11),
        F(9, 22),
        F(-2, 33),
        F(0),
        F(0),
        F(18, 11),
        F(-9, 22),
        F(2, 33),
    ),
    (
        F(360, 239),
        F(126, 239),
        F(-88, 717),
        F(-405, 239),
        F(80, 239),
        F(360, 239),
        F(126, 239),
        F(-88, 717),
    ),
    (
        F(-76, 239),
        F(547, 239),
        F(-964, 2151),
        F(-990, 239),
        F(1760, 2151),
        F(-76, 239),
        F(547, 239),
        F(-964, 2151),
    ),
    (
        F(-760, 717),
        F(1168, 717),
        F(-664, 2151),
        F(-432, 239),
        F(256, 717),
        F(-760, 717),
        F(1168, 717),
        F(-664, 2151),
    ),
    (
        F(-76, 33),
        F(52, 33),
        F(-28, 99),
        F(0),
        F(0),
        F(76, 33),
        F(-52, 33),
        F(28, 99),
    ),
    (
        F(-3056, 10755),
        F(476, 1195),
        F(-2992, 32265),
        F(-1244, 3585),
        F(808, 10755),
        F(-3056, 10755),
        F(476, 1195),
        F(-2992, 32265),
    ),
    (
        F(-40, 33),
        F(32, 33),
        F(-8, 33),
        F(0),
        F(0),
        F(40, 33),
        F(-32, 33),
        F(8, 33),
    ),
    (
        F(-472, 10755),
        F(646, 3585),
        F(-2264, 32265),
        F(-1018, 3585),
        F(1276, 10755),
        F(-472, 10755),
        F(646, 3585),
        F(-2264, 32265),
    ),
)

EXPECTED_INVERSE_ROW_SUMS = (
    F(139, 33),
    F(4547, 717),
    F(23812, 2151),
    F(17552, 2151),
    F(824, 99),
    F(63644, 32265),
    F(160, 33),
    F(31978, 32265),
)

EXPECTED_ROW_DEGREE_SUMS = (
    (F(225, 128), F(189, 256), F(47, 128)),
    (F(135, 8), F(165, 16), F(47, 8)),
    (F(9369, 128), F(12717, 256), F(3807, 128)),
    (F(18), F(8), F(2)),
    (F(315, 4), F(297, 8), F(81, 8)),
    (F(225, 128), F(189, 256), F(47, 128)),
    (F(135, 8), F(165, 16), F(47, 8)),
    (F(9369, 128), F(12717, 256), F(3807, 128)),
)

EXPECTED_RADIUS_TABLE = (
    (F(1, 2), F(3375, 64), F(2232375, 3824), False),
    (F(1, 4), F(11349, 512), F(7506733, 30592), False),
    (F(1, 8), F(42777, 4096), F(28294609, 244736), False),
    (F(1, 16), F(166113, 32768), F(109874521, 1957888), False),
    (F(1, 32), F(654705, 262144), F(433050985, 15663104), False),
    (F(1, 64), F(2599569, 2097152), F(1719470473, 125304832), False),
    (F(1, 128), F(10360017, 16777216), F(6852575689, 1002438656), False),
    (F(1, 256), F(41363793, 134217728), F(27359851081, 8019509248), False),
    (F(1, 512), F(165302865, 1073741824), F(109338661705, 64156073984), False),
    (F(1, 1024), F(660907089, 8589934592), F(437153322313, 513248591872), False),
    (F(1, 2048), F(2643019857, 68719476736), F(1748210800969, 4105988734976), True),
    (F(1, 4096), F(10570862673, 549755813888), F(6992038388041, 32847909879808), True),
    (F(1, 8192), F(42281017425, 4398046511104), F(27966544081225, 262783279038464), True),
    (F(1, 16384), F(169119203409, 35184372088832), F(111862957543753, 2102266232307712), True),
    (F(1, 32768), F(676467081297, 281474976710656), F(447445392773449, 16818129858461696), True),
    (
        F(1, 65536),
        F(2705848860753, 2251799813685248),
        F(1789768696451401, 134545038867693568),
        True,
    ),
)

SELECTED_RADIUS = F(1, 2048)
SELECTED_DELTA = F(2643019857, 68719476736)
SELECTED_PRODUCT = F(1748210800969, 4105988734976)
LOWER_LIPSCHITZ = F(21220001406063, 409087045009408)
INVERSE_LIPSCHITZ = F(409087045009408, 21220001406063)
INVERSE_NORM = F(23812, 2151)


def _blocks(coordinates: tuple[Fraction, ...]):
    return (
        coordinates[0:2],
        coordinates[2:3],
        coordinates[3:5],
        coordinates[5:8],
    )


def _infinity_norm(values: tuple[Fraction, ...]) -> Fraction:
    return max(abs(value) for value in values)


def _scaled_vector(
    multipliers: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(SELECTED_RADIUS * value for value in multipliers)


def test_exact_response_and_jacobian_polynomial_coefficients() -> None:
    response = exact_centered_observation_polynomials(DESIGN)
    jacobian_difference = exact_jacobian_difference_polynomials(DESIGN)

    e1 = (1, 0, 0, 0, 0, 0, 0, 0)
    z1_fourth = (4, 0, 0, 0, 0, 0, 0, 0)
    z1_z2_z3 = (1, 1, 1, 0, 0, 0, 0, 0)
    z1_z2_cubed = (1, 3, 0, 0, 0, 0, 0, 0)

    assert tuple(len(polynomial) for polynomial in response) == (
        27,
        27,
        27,
        14,
        14,
        27,
        27,
        27,
    )
    assert response[0][e1] == F(-1, 2)
    assert response[0][z1_fourth] == F(17, 1024)
    assert response[0][z1_z2_z3] == F(-7, 64)
    assert response[4][z1_z2_cubed] == F(-27, 32)

    assert jacobian_difference[0][0][(3, 0, 0, 0, 0, 0, 0, 0)] == F(17, 256)
    assert jacobian_difference[0][2][e1] == F(-3, 64)
    assert jacobian_difference[4][0][(0, 3, 0, 0, 0, 0, 0, 0)] == F(-27, 32)
    assert jacobian_difference[4][1][(1, 2, 0, 0, 0, 0, 0, 0)] == F(-81, 32)
    assert exact_jacobian_row_degree_sums(DESIGN) == EXPECTED_ROW_DEGREE_SUMS


def test_exact_inverse_row_sums_and_infinity_norm() -> None:
    _, determinant, inverse = certify_exact_design(DESIGN)

    assert determinant == F(-258733035, 524288)
    assert inverse == EXPECTED_INVERSE
    assert exact_inverse_row_sums(inverse) == EXPECTED_INVERSE_ROW_SUMS
    assert exact_matrix_infinity_norm(inverse) == INVERSE_NORM
    assert max(EXPECTED_INVERSE_ROW_SUMS) == INVERSE_NORM


def test_complete_exact_radius_table_and_selected_constants() -> None:
    assert stability_test_radii() == tuple(entry[0] for entry in EXPECTED_RADIUS_TABLE)
    assert exact_stability_radius_table(DESIGN) == EXPECTED_RADIUS_TABLE
    assert exact_jacobian_variation_bound(DESIGN, SELECTED_RADIUS) == SELECTED_DELTA
    assert exact_local_stability_constants(DESIGN) == (
        SELECTED_RADIUS,
        SELECTED_DELTA,
        SELECTED_PRODUCT,
        LOWER_LIPSCHITZ,
        INVERSE_LIPSCHITZ,
        INVERSE_NORM,
    )
    assert EXPECTED_RADIUS_TABLE[9][3] is False
    assert EXPECTED_RADIUS_TABLE[10][3] is True
    selected_index = next(
        index for index, entry in enumerate(EXPECTED_RADIUS_TABLE) if entry[3]
    )
    assert EXPECTED_RADIUS_TABLE[selected_index][0] == SELECTED_RADIUS
    assert all(not entry[3] for entry in EXPECTED_RADIUS_TABLE[:selected_index])
    assert max(entry[0] for entry in EXPECTED_RADIUS_TABLE if entry[3]) == (
        SELECTED_RADIUS
    )
    assert SELECTED_PRODUCT <= F(1, 2)
    assert LOWER_LIPSCHITZ == 1 / INVERSE_NORM - SELECTED_DELTA
    assert INVERSE_LIPSCHITZ == 1 / LOWER_LIPSCHITZ
    assert INVERSE_LIPSCHITZ < 2 * INVERSE_NORM


@pytest.mark.parametrize("multiplier", (F(2), F(-2), F(1, 2), F(-1, 2)))
def test_common_scale_inverse_factorization(multiplier: Fraction) -> None:
    scaled_design = common_rescale_design(DESIGN, multiplier)
    _, _, directly_inverted = certify_exact_design(scaled_design)

    assert directly_inverted == exact_common_scale_inverse(
        EXPECTED_INVERSE,
        multiplier,
    )


@pytest.mark.parametrize(
    ("multiplier", "expected_norm"),
    (
        (F(2), F(4547, 1434)),
        (F(-2), F(4547, 1434)),
        (F(1, 2), F(2560, 33)),
        (F(-1, 2), F(2560, 33)),
    ),
)
def test_common_scale_inverse_row_sum_formula_and_degree_four_exponent(
    multiplier: Fraction,
    expected_norm: Fraction,
) -> None:
    expected_from_rows = max(
        abs(multiplier) ** (-degree) * row_sum
        for degree, row_sum in zip(
            hall_basis_degrees(),
            EXPECTED_INVERSE_ROW_SUMS,
            strict=True,
        )
    )
    factorized = exact_common_scale_inverse(EXPECTED_INVERSE, multiplier)

    assert hall_basis_degrees() == (1, 1, 2, 3, 3, 4, 4, 4)
    assert (
        exact_common_scale_inverse_norm(EXPECTED_INVERSE, multiplier)
        == expected_norm
    )
    assert expected_from_rows == expected_norm
    assert exact_matrix_infinity_norm(factorized) == expected_norm
    if abs(multiplier) == F(1, 2):
        assert expected_norm == abs(multiplier) ** (-4) * F(160, 33)


def test_common_scale_inverse_amplification_has_exact_fourth_order() -> None:
    norm_at_half = exact_common_scale_inverse_norm(EXPECTED_INVERSE, F(1, 2))
    norm_at_quarter = exact_common_scale_inverse_norm(EXPECTED_INVERSE, F(1, 4))
    assert norm_at_quarter == F(2) ** 4 * norm_at_half
    assert norm_at_quarter == F(1, 4) ** (-4) * F(160, 33)


STABILITY_PAIRS = (
    (
        _scaled_vector(
            (
                F(1, 4),
                F(-1, 5),
                F(1, 6),
                F(-1, 7),
                F(1, 8),
                F(-1, 9),
                F(1, 10),
                F(-1, 11),
            )
        ),
        _scaled_vector(
            (
                F(-1, 6),
                F(1, 7),
                F(-1, 8),
                F(1, 9),
                F(-1, 10),
                F(1, 11),
                F(-1, 12),
                F(1, 13),
            )
        ),
    ),
    (
        _scaled_vector(
            (
                F(2047, 2048),
                F(-2045, 2048),
                F(7, 8),
                F(-6, 7),
                F(5, 6),
                F(-4, 5),
                F(3, 4),
                F(-2, 3),
            )
        ),
        _scaled_vector(
            (
                F(-2043, 2048),
                F(2041, 2048),
                F(-11, 12),
                F(10, 11),
                F(-9, 10),
                F(8, 9),
                F(-7, 8),
                F(6, 7),
            )
        ),
    ),
    (
        _scaled_vector(
            (
                F(15, 16),
                F(-14, 15),
                F(13, 14),
                F(-12, 13),
                F(11, 12),
                F(-10, 11),
                F(9, 10),
                F(-8, 9),
            )
        ),
        _scaled_vector(
            (
                F(-15, 16),
                F(14, 15),
                F(-13, 14),
                F(12, 13),
                F(-11, 12),
                F(10, 11),
                F(-9, 10),
                F(8, 9),
            )
        ),
    ),
)


@pytest.mark.parametrize(("left", "right"), STABILITY_PAIRS)
def test_exact_rational_local_stability_guards(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> None:
    assert all(value != 0 for value in left + right)
    assert _infinity_norm(left) < SELECTED_RADIUS
    assert _infinity_norm(right) < SELECTED_RADIUS

    left_observation = exact_centered_truncated_observation_map(
        _blocks(left),
        DESIGN,
    )
    right_observation = exact_centered_truncated_observation_map(
        _blocks(right),
        DESIGN,
    )
    coordinate_distance = _infinity_norm(
        tuple(a - b for a, b in zip(left, right, strict=True))
    )
    observation_distance = _infinity_norm(
        tuple(
            a - b
            for a, b in zip(
                left_observation,
                right_observation,
                strict=True,
            )
        )
    )

    assert coordinate_distance > 0
    assert LOWER_LIPSCHITZ * coordinate_distance <= observation_distance
    assert coordinate_distance <= INVERSE_LIPSCHITZ * observation_distance


def _finite_certificate_payload() -> dict:
    path = ROOT / "research_output" / "finite_observation_certificate.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_finite_certificate_observation_order_mutation_fails() -> None:
    payload = deepcopy(_finite_certificate_payload())
    payload["selected_design"][0], payload["selected_design"][1] = (
        payload["selected_design"][1],
        payload["selected_design"][0],
    )
    with pytest.raises(RuntimeError, match="matrix is inconsistent"):
        _validated_finite_certificate(payload)


def test_finite_certificate_determinant_mutation_fails() -> None:
    payload = deepcopy(_finite_certificate_payload())
    payload["zero_jacobian"]["exact_determinant"] = "1"
    with pytest.raises(RuntimeError, match="determinant is inconsistent"):
        _validated_finite_certificate(payload)


def test_reproducer_verifies_certificate_without_writing(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research_output" / "reproduce_local_stability.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "local_stability_certificate: verified" in completed.stdout
    assert tuple(tmp_path.iterdir()) == before

"""Run the small exact tensor-witness checks without writing files."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lie_reservoir.tensor_witness import (  # noqa: E402
    BLOCK_EMBEDDING_CASES,
    MAX_REPRODUCER_STATE_DIM,
    STRUCTURAL_CASES,
    TANH_LIMIT_DIAGNOSTIC_CASES,
    block_embedding_checks,
    hall_convention_checks,
    length_cutoff_holds,
    noncertifying_normalized_tanh_limit_coefficients,
    nonempty_tensor_words,
    signature_convention_checks,
    tensor_state_dimension,
    word_evaluation_matrix,
)


def _integer_identity(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(row == column) for column in range(size))
        for row in range(size)
    )


def _fraction_identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(size))
        for row in range(size)
    )


def _verify_case(input_dim: int, degree: int) -> str:
    state_dim = tensor_state_dimension(input_dim, degree)
    if state_dim > MAX_REPRODUCER_STATE_DIM:
        raise RuntimeError(
            f"D_0={state_dim} exceeds the reproducer cap "
            f"{MAX_REPRODUCER_STATE_DIM}."
        )
    word_count = len(nonempty_tensor_words(input_dim, degree))
    a0_ok = word_evaluation_matrix(input_dim, degree) == _integer_identity(
        word_count
    )
    cutoff_ok = length_cutoff_holds(input_dim, degree)
    one_segment_ok, two_segment_ok = signature_convention_checks(
        input_dim,
        degree,
    )
    hall_equal, hall_rank, hall_columns = hall_convention_checks(
        input_dim,
        degree,
    )
    hall_ok = hall_equal and hall_rank == hall_columns

    if (input_dim, degree) in BLOCK_EMBEDDING_CASES:
        block_ok = all(block_embedding_checks(input_dim, degree))
        block_status = "pass"
    else:
        block_ok = True
        block_status = "n/a"

    if (input_dim, degree) in TANH_LIMIT_DIAGNOSTIC_CASES:
        try:
            constant, linear = noncertifying_normalized_tanh_limit_coefficients(
                input_dim,
                degree,
            )
        except RuntimeError as error:
            if "SymPy is unavailable" not in str(error):
                raise
            tanh_ok = True
            tanh_status = "unavailable(non-certifying)"
        else:
            tanh_ok = constant == _fraction_identity(word_count) and all(
                value == 0 for row in linear for value in row
            )
            tanh_status = "pass(non-certifying)"
    else:
        tanh_ok = True
        tanh_status = "n/a(non-certifying)"

    statuses = (
        a0_ok,
        cutoff_ok,
        one_segment_ok,
        two_segment_ok,
        hall_ok,
        block_ok,
        tanh_ok,
    )
    if not all(statuses):
        raise RuntimeError(
            "an exact tensor-witness check failed for "
            f"input_dim={input_dim}, degree={degree}: {statuses}."
        )

    return (
        f"input_dim={input_dim} degree={degree} D_0={state_dim} "
        f"words={word_count} A_0=identity cutoff=pass "
        f"signature=pass(one,two) Hall-rank=pass({hall_rank}/{hall_columns}) "
        f"block={block_status} tanh-limit={tanh_status}"
    )


def main() -> int:
    for input_dim, degree in STRUCTURAL_CASES:
        print(_verify_case(input_dim, degree))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

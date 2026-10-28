"""Deterministic high-precision checks for the fixed eight-response decoder.

Run from the repository root. Prints results only; writes no artifacts.
Exact tensor/algebra checks precede numerical flow and fixed-point checks.
Numerical residuals are not interval-certified residual bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lie_reservoir.finite_observation import (  # noqa: E402
    common_rescale_design,
    exact_centered_observation_polynomials,
    exact_local_stability_constants,
    hall_basis_degrees,
)
from lie_reservoir.free_lie import lyndon_basis, lyndon_words, word_expansion  # noqa: E402
from lie_reservoir.graded_reconstruction import (  # noqa: E402
    _word_operator_jet,
    tensor_add,
    tensor_exp,
    truncated_tensor_product,
)
from research_output.reproduce_local_stability import (  # noqa: E402
    _validated_finite_certificate,
)

DEGREES = hall_basis_degrees()
ASYMMETRIC = ((0, F(1, 3)), (1, F(1, 4)), (0, F(-1, 5)), (1, F(1, 6)))
PATHS = {
    "alternating": ASYMMETRIC,
    "rectangle": ((0, F(1, 3)), (1, F(1, 4)), (0, F(-1, 3)), (1, F(-1, 4))),
    "reversed": tuple((axis, -a) for axis, a in reversed(ASYMMETRIC)),
}


def exact_path_hall(segments):
    """Chen product in path order, finite tensor log, existing Hall expansion."""
    signature = {(): F(1)}
    for axis, increment in segments:
        signature = truncated_tensor_product(
            signature, tensor_exp({(axis,): increment})
        )
    positive = {word: value for word, value in signature.items() if word}
    power, logarithm = {(): F(1)}, {}
    for k in range(1, 5):
        power = truncated_tensor_product(power, positive)
        logarithm = tensor_add(
            logarithm, {word: F((-1) ** (k + 1), k) * a for word, a in power.items()}
        )
    hall, rebuilt = [], {}
    for degree in range(1, 5):
        pivots, nodes = lyndon_words(2, degree), lyndon_basis(2, degree)
        columns = [dict(word_expansion(node)) for node in nodes]
        assert [[column.get(w, 0) for column in columns] for w in pivots] == [
            [int(i == j) for j in range(len(nodes))] for i in range(len(nodes))
        ]
        for word, column in zip(pivots, columns, strict=True):
            value = logarithm.get(word, F(0))
            hall.append(value)
            rebuilt = tensor_add(rebuilt, {w: value * a for w, a in column.items()})
    assert rebuilt == logarithm
    assert tensor_exp(logarithm) == signature
    return tuple(hall)


def exact_setup():
    design, matrix, _, inverse = _validated_finite_certificate()
    polynomials = exact_centered_observation_polynomials(design)
    radius, variation, kappa, _, constant, inverse_norm = exact_local_stability_constants(design)
    stored = json.loads((ROOT / "research_output/local_stability_certificate.json").read_text())
    assert radius == F(stored["radius_search"]["selected_radius"])
    assert inverse_norm == F(stored["zero_jacobian"]["exact_inverse_infinity_norm"])
    assert constant == F(stored["stability_constants"]["exact_inverse_lipschitz_bound"])
    assert kappa == inverse_norm * variation < F(1, 2)
    assert constant == inverse_norm / (1 - kappa)
    assert stored["selected_design"] == [
        {"state": state, "scale": str(scale)} for state, scale in design
    ]
    assert max(abs(scale) for _, scale in design) == 3
    for h in (F(1, 2), F(-2), F(1, 2048)):
        scaled = exact_centered_observation_polynomials(common_rescale_design(design, h))
        for original, actual in zip(polynomials, scaled, strict=True):
            assert actual == {
                alpha: coefficient * h ** sum(a * d for a, d in zip(alpha, DEGREES))
                for alpha, coefficient in original.items()
            }
    # Apply the manuscript's existing outermost-first polynomial recursion.
    s = sp.Symbol("s")
    fields = (s, sp.Rational(1, 2))
    words = {(0,): fields[0], (1,): fields[1]}
    for length in range(2, 6):
        words = {
            (i,) + word: sp.Poly(fields[i] * (1 - s**2) * sp.diff(p, s), s).as_expr()
            for i in range(2) for word, p in words.items()
        }
        if length == 5:
            for word, p in words.items():
                for state in (F(-1, 2), F(0), F(1, 2)):
                    assert F(p.subs(s, sp.Rational(state.numerator, state.denominator))) == (
                        _word_operator_jet(word, (state, F(1, 2)), (F(1), F(0)), 0)[0]
                    )
    coefficient_bound = max(sum(abs(a) for a in sp.Poly(p, s).all_coeffs()) for p in words.values())
    assert coefficient_bound <= 1680
    hall = {name: exact_path_hall(path) for name, path in PATHS.items()}
    assert hall["reversed"] == tuple(-a for a in hall["alternating"])
    c = (F(1), F(1), F(1), F(13, 12))
    for name, path in PATHS.items():
        mass = sum(abs(a) for _, a in path)
        assert all(abs(a) <= c[d - 1] * mass**d for a, d in zip(hall[name], DEGREES))
    print(f"Exact checks passed; R={radius}, K={inverse_norm}, kappa={kappa}, C={constant}.")
    print(f"Length-five coefficient norm maximum={coefficient_bound}; theorem uses K5=1680, A=3402.")
    return design, polynomials, matrix, inverse, radius, kappa, constant, hall


def number(value):
    if isinstance(value, F):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def norm(vector):
    return max(abs(a) for a in vector)


def polynomial_value(polynomials, u):
    powers = [[mp.mpf(1), a, a**2, a**3, a**4] for a in u]
    return mp.matrix([
        mp.fsum(coefficient * mp.fprod(powers[i][a] for i, a in enumerate(alpha))
                for alpha, coefficient in polynomial.items())
        for polynomial in polynomials
    ])


def clipped_step(u, b, polynomials, inverse, radius):
    candidate = u + inverse * (b - polynomial_value(polynomials, u))
    return mp.matrix([min(radius, max(-radius, a)) for a in candidate])


def decode(b, polynomials, inverse, radius, kappa, h, accuracy):
    """Fixed eight-coordinate clipped Picard iteration; tolerance is unscaled."""
    u = mp.matrix(8, 1)
    tolerance = (1 - kappa) * h**4 * accuracy
    for iteration in range(500):
        following = clipped_step(u, b, polynomials, inverse, radius)
        residual = norm(u - following)
        if residual <= tolerance:
            return u, residual, iteration
        u = following
    raise RuntimeError("Clipped iteration did not reach its scaled residual tolerance.")


def actual_responses(segments, design, h):
    """Chronological axis flows: the last segment acts outermost on the state."""
    q = mp.log(3) / 2
    states = {"-log(3)/2": -q, "0": mp.mpf(0), "log(3)/2": q}
    values = []
    for label, scale in design:
        initial = states[label]
        y = initial
        for axis, increment in segments:
            a = h * number(scale) * number(increment)
            y = mp.asinh(mp.exp(a) * mp.sinh(y)) if axis == 0 else y + a / 2
        values.append(y - initial)
    return mp.matrix(values)


def numerical_checks(setup, dps):
    design, exact_polynomials, exact_matrix, exact_inverse, exact_radius, exact_kappa, exact_c, hall = setup
    with mp.workdps(dps):
        radius, kappa, constant = map(number, (exact_radius, exact_kappa, exact_c))
        inverse = mp.matrix([[number(a) for a in row] for row in exact_inverse])
        matrix = mp.matrix([[number(a) for a in row] for row in exact_matrix])
        polynomials = [{alpha: number(a) for alpha, a in p.items()} for p in exact_polynomials]
        accuracy = mp.power(10, -(dps - 25))
        points = [mp.matrix(8, 1), mp.matrix([radius] * 8), mp.matrix([-radius] * 8),
                  mp.matrix([(-1)**i * radius for i in range(8)]),
                  mp.matrix([radius, -radius / 2, 0, radius / 3, 0, 0, 0, 0])]
        roundtrip_error = mp.mpf(0)
        for target in points:
            b = polynomial_value(polynomials, target)
            u, residual, _ = decode(b, polynomials, inverse, radius, kappa, mp.mpf(1), accuracy)
            roundtrip_error = max(roundtrip_error, norm(u - target))
            assert norm(u - target) <= 2 * accuracy
        # Deliberately off-image: the upper corner is a clipped fixed point.
        corner = points[1]
        b_outside = polynomial_value(polynomials, corner) + matrix * corner
        u, clipped_residual, _ = decode(b_outside, polynomials, inverse, radius, kappa, mp.mpf(1), accuracy)
        clipped_fit = norm(polynomial_value(polynomials, u) - b_outside)
        assert norm(u - corner) <= 2 * accuracy and clipped_fit > radius
        results = []
        for name, segments in PATHS.items():
            mass_exact = sum(abs(a) for _, a in segments)
            mass = number(mass_exact)
            z = mp.matrix([number(a) for a in hall[name]])
            for divisor in (1, 2, 4):
                h = number(exact_radius / (mass_exact * divisor))
                scaled = mp.matrix([a * h**d for a, d in zip(z, DEGREES)])
                assert h <= 1 and norm(scaled) <= radius
                actual = actual_responses(segments, design, h)
                truncated = polynomial_value(polynomials, scaled)
                remainder = norm(actual - truncated)
                assert remainder <= 3402 * h**5 * mass**5
                for noise in (False, True):
                    delta = (h * mass)**5 / 100 if noise else mp.mpf(0)
                    perturbation = mp.matrix([delta * (-1)**i for i in range(8)])
                    assert norm(perturbation) == delta
                    b = actual + perturbation
                    u, residual, iterations = decode(b, polynomials, inverse, radius, kappa, h, accuracy)
                    estimate = mp.matrix([a / h**d for a, d in zip(u, DEGREES)])
                    error = norm(estimate - z)
                    fit = norm(polynomial_value(polynomials, u) - b)
                    bound = constant * (3402 * mass**5 * h + delta / h**4)
                    assert error <= bound + residual / ((1 - kappa) * h**4)
                    results.append(dict(path=name, divisor=divisor, noise=noise, h=h,
                                        delta=delta, remainder=remainder, error=error,
                                        residual=residual, fit=fit, bound=bound,
                                        iterations=iterations, estimate=tuple(estimate)))
        print(f"{dps} dps: five polynomial round trips max error={mp.nstr(roundtrip_error, 6)}; "
              f"off-image fixed residual={mp.nstr(clipped_residual, 6)}, fit residual={mp.nstr(clipped_fit, 6)}.")
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--verify-dps", type=int, default=120)
    args = parser.parse_args()
    if args.dps < 60 or args.verify_dps < args.dps + 20:
        parser.error("use at least 60 dps and a verification precision at least 20 digits higher")
    setup = exact_setup()
    first = numerical_checks(setup, args.dps)
    higher = numerical_checks(setup, args.verify_dps)
    with mp.workdps(args.verify_dps):
        disagreement = max(abs(a - b) for low, high in zip(first, higher, strict=True)
                           for a, b in zip(low["estimate"], high["estimate"], strict=True))
        assert disagreement < mp.power(10, -(args.dps - 30))
        print(f"Maximum coordinate disagreement across precisions: {mp.nstr(disagreement, 8)}")
        print("Numerical evidence only; finite-precision residuals are not exact certificates.")
        print("| Path | h | delta | Response remainder | Coordinate error | Fixed residual | Fit residual | Data + truncation bound | Iterations |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in first:
            keys = ("h", "delta", "remainder", "error", "residual", "fit", "bound")
            print("| " + row["path"] + " | " + " | ".join(mp.nstr(row[key], 7) for key in keys)
                  + f" | {row['iterations']} |")


if __name__ == "__main__":
    main()

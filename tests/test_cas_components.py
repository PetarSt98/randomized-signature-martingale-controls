"""Fixed-premium and frozen-field invariants for the bounded CAS comparison."""

from __future__ import annotations

import numpy as np
import pytest

from rsmc.features import PathFeatures
from rsmc.hedging import (
    assemble_reservoir_hedge_features,
    discounted_traded_increments,
    evaluate_hedge,
    fit_fixed_premium_hedge,
    hedging_features,
)
from rsmc.reservoir import DepthTwoSignatureReservoir, RandomizedSignatureReservoir


def test_depth_two_matches_explicit_recurrence_and_starts_moving() -> None:
    reservoir = DepthTwoSignatureReservoir.draw(2, 3, np.random.default_rng(61), 0.7, 0.4)
    features = PathFeatures(
        np.array([[[0.0, 0.0], [0.2, -0.1], [0.3, 0.15], [0.5, 0.0]]]),
        ("time", "state"),
    )
    states = reservoir.run(features)
    expected = np.zeros_like(states)
    for k in range(3):
        previous = expected[0, k]
        fields = []
        for channel in range(2):
            inner = np.tanh(reservoir.weights[channel] @ previous + reservoir.biases[channel])
            fields.append(np.tanh(reservoir.outer_weights[channel] @ inner + reservoir.outer_biases[channel]))
        expected[0, k + 1] = previous + features.increments[0, k] @ np.asarray(fields)
    np.testing.assert_allclose(states, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(states[:, 0], 0.0)
    assert np.linalg.norm(states[:, 1]) > 0
    assert reservoir.weights.shape == reservoir.outer_weights.shape == (2, 3, 3)
    repeat = DepthTwoSignatureReservoir.draw(2, 3, np.random.default_rng(61), 0.7, 0.4)
    np.testing.assert_array_equal(states, repeat.run(features))


@pytest.mark.parametrize("kind", [RandomizedSignatureReservoir, DepthTwoSignatureReservoir])
def test_causal_gains_and_frozen_parameters_through_fit(kind) -> None:
    times = np.linspace(0.0, 1.0, 5)
    spot = np.array([[100, 103, 102, 106, 109], [100, 97, 99, 96, 95],
                     [100, 101, 105, 104, 102], [100, 98, 101, 103, 104]], dtype=float)
    features = hedging_features(spot, times)
    increments = discounted_traded_increments(spot, times, 0.03)
    reservoir = kind.draw(features.n_channels, 3, np.random.default_rng(62))
    arrays = {name: value.copy() for name, value in vars(reservoir).items() if isinstance(value, np.ndarray)}
    states = reservoir.run(features)
    altered = features.values.copy()
    altered[:, 3:] += 10.0
    altered_states = reservoir.run(PathFeatures(altered, features.names))
    np.testing.assert_array_equal(states[:, :3], altered_states[:, :3])
    psi = assemble_reservoir_hedge_features(reservoir, features, times, increments)
    beta = np.arange(psi.shape[1], dtype=float) / 10
    explicit_gain = np.zeros(spot.shape[0])
    for k in range(len(times) - 1):
        phi = np.column_stack([np.ones(4), np.full(4, times[k]), features.values[:, k, 1:], states[:, k]])
        explicit_gain += (phi @ beta) * increments[:, k]
    np.testing.assert_allclose(psi @ beta, explicit_gain)
    # The terminal feature is not available to any pre-terminal trading decision.
    terminal_changed = features.values.copy()
    terminal_changed[:, -1] += 1000
    same_psi = assemble_reservoir_hedge_features(
        reservoir, PathFeatures(terminal_changed, features.names), times, increments,
    )
    np.testing.assert_array_equal(psi, same_psi)
    fitted = fit_fixed_premium_hedge(psi, np.array([4.0, 0.0, 2.0, 3.0]), 2.25, 1.0)
    evaluate_hedge(fitted, psi, np.array([4.0, 0.0, 2.0, 3.0]))
    for name, before in arrays.items():
        np.testing.assert_array_equal(getattr(reservoir, name), before)


@pytest.mark.parametrize("shape,lam", [((12, 4), 0.0), ((12, 4), 2.0), ((4, 12), 0.0), ((4, 12), 2.0)])
def test_fixed_premium_ridge_matches_augmented_least_squares(shape, lam) -> None:
    rng = np.random.default_rng(63)
    design = rng.normal(size=shape) + 2.0  # Nonzero means expose accidental intercept fitting.
    design[:, -1] = 0.0                  # Constant-zero columns must remain harmless.
    payoff = rng.normal(size=shape[0]) + 7.0
    premium = 3.25
    fitted = fit_fixed_premium_hedge(design, payoff, premium, lam)
    scale = np.sqrt(np.mean(design**2, axis=0))
    scale[scale < 1e-12] = 1.0
    reference, *_ = np.linalg.lstsq(
        np.vstack([design / scale, np.sqrt(lam) * np.eye(shape[1])]),
        np.concatenate([payoff - premium, np.zeros(shape[1])]), rcond=None,
    )
    np.testing.assert_allclose(fitted.beta, reference / scale, rtol=1e-10, atol=1e-10)
    assert fitted.intercept == premium
    test_rows = rng.normal(size=(8, shape[1])) + 20.0
    test_payoff = rng.normal(size=8) + 12.0
    report = evaluate_hedge(fitted, test_rows, test_payoff)
    expected_errors = test_payoff - premium - test_rows @ fitted.beta
    np.testing.assert_allclose(report.errors, expected_errors)
    assert report.error_mse == pytest.approx(np.mean(expected_errors**2))
    assert abs(report.mean_error) > 0.1  # No test-error recentering.

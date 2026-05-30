"""Tests for the ridge readout and the martingale control variate.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_control.py``).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    BlackScholesModel,
    RandomizedSignatureReservoir,
    RidgeMartingaleControl,
    arithmetic_asian_call,
    asian_features,
    assemble_control_features,
    control_features,
    evaluate_control,
    geometric_asian_call_price,
    readout_features,
    tune_ridge_lambda,
    variance_reduction,
)

MODEL = BlackScholesModel(s0=100.0, r=0.03, sigma=0.20, T=1.0)
STRIKE = 100.0


@lru_cache(maxsize=None)
def _build(n_train=6_000, n_test=20_000, n_steps=50, seed=0):
    """Build Phi/G for train and test (cached so repeated calls are cheap)."""
    train_seed, test_seed, draw_seed = np.random.SeedSequence(seed).spawn(3)
    reservoir = RandomizedSignatureReservoir.draw(
        n_channels=3, dim=64, rng=np.random.default_rng(draw_seed)
    )

    def rows(n_paths, rng_seed):
        paths = MODEL.simulate(n_paths, n_steps, np.random.default_rng(rng_seed))
        features = asian_features(paths)
        phi = assemble_control_features(
            reservoir, features, paths.times, paths.brownian_increments
        )
        g = arithmetic_asian_call(paths, strike=STRIKE, rate=MODEL.r)
        return phi, g

    phi_train, g_train = rows(n_train, train_seed)
    phi_test, g_test = rows(n_test, test_seed)
    return phi_train, g_train, phi_test, g_test


def test_control_features_shape_matches_readout() -> None:
    """Phi has shape (n, D + 2) and equals the manual einsum."""
    paths = MODEL.simulate(500, 50, np.random.default_rng(1))
    features = asian_features(paths)
    reservoir = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(2))
    phi = assemble_control_features(
        reservoir, features, paths.times, paths.brownian_increments
    )
    assert phi.shape == (500, 66)

    states = reservoir.run(features)
    manual = control_features(readout_features(states, paths.times), paths.brownian_increments)
    assert np.allclose(phi, manual)


def test_controlled_estimator_is_unbiased() -> None:
    """mean(C) on independent test paths is zero within Monte Carlo noise."""
    phi_train, g_train, phi_test, g_test = _build()
    control = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)
    c = control.control(phi_test)
    stderr = c.std(ddof=1) / np.sqrt(c.size)
    assert abs(c.mean()) < 5.0 * stderr


def test_variance_reduction_above_one() -> None:
    """The fitted control reduces variance out of sample (VR > 1)."""
    phi_train, g_train, phi_test, g_test = _build()
    control = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)
    assert variance_reduction(control, phi_test, g_test) > 1.5


def test_price_estimates_agree() -> None:
    """Plain MC and controlled price agree within a few standard errors."""
    phi_train, g_train, phi_test, g_test = _build()
    control = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)
    report = evaluate_control(control, phi_test, g_test)
    assert abs(report.price_mc - report.price_cv) < 5.0 * report.se_mc
    # The controlled estimator must have the smaller standard error.
    assert report.se_cv < report.se_mc


def test_report_is_internally_consistent() -> None:
    """The reported metrics satisfy their defining identities."""
    phi_train, g_train, phi_test, g_test = _build()
    control = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)
    report = evaluate_control(control, phi_test, g_test)
    assert np.isclose(report.se_mc, np.sqrt(report.var_mc / report.n_eval))
    assert np.isclose(report.variance_reduction, report.var_mc / report.var_cv)
    assert np.isclose(report.price_cv, report.price_mc - report.control_mean, atol=1e-9)


def test_fit_is_reproducible_and_lambda_matters() -> None:
    """Same data and lambda -> identical beta; larger lambda -> smaller ||beta||."""
    phi_train, g_train, _, _ = _build(n_train=4_000, n_test=1_000)
    a = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)
    b = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)
    big = RidgeMartingaleControl.fit(phi_train, g_train, lam=1e6)
    assert np.array_equal(a.beta, b.beta)
    assert np.linalg.norm(big.beta) < np.linalg.norm(a.beta)


def test_tune_lambda_picks_from_grid() -> None:
    """Lambda tuning returns a grid value and a full diagnostic table."""
    phi_train, g_train, phi_val, g_val = _build(n_train=4_000, n_test=4_000)
    grid = (0.0, 10.0, 1_000.0)
    best, table = tune_ridge_lambda(phi_train, g_train, phi_val, g_val, grid)
    assert best in grid
    assert len(table) == len(grid)


def test_geometric_price_is_lower_bound() -> None:
    """Arithmetic Asian MC price should sit just above the geometric price."""
    paths = MODEL.simulate(100_000, 50, np.random.default_rng(7))
    arithmetic = arithmetic_asian_call(paths, strike=STRIKE, rate=MODEL.r).mean()
    geometric = geometric_asian_call_price(
        s0=MODEL.s0, strike=STRIKE, rate=MODEL.r, sigma=MODEL.sigma,
        maturity=MODEL.T, n_steps=50,
    )
    assert 0.0 < geometric < arithmetic
    assert arithmetic - geometric < 0.5      # they are close for these parameters


def test_vector_control_features_match_manual_blocks() -> None:
    """Vector increments give (n, m*q) = one manually-summed block per direction."""
    rng = np.random.default_rng(3)
    n, n_steps, q, m = 40, 12, 5, 2
    readout_phi = rng.standard_normal((n, n_steps + 1, q))
    increments = rng.standard_normal((n, n_steps, m))
    phi = control_features(readout_phi, increments)
    assert phi.shape == (n, m * q)
    predictable = readout_phi[:, :-1, :]
    for j in range(m):                       # block j = sum_k phi_k * dM^{(j)}_{k+1}
        block = np.einsum("nkq,nk->nq", predictable, increments[:, :, j])
        assert np.allclose(phi[:, j * q:(j + 1) * q], block)


def test_single_direction_vector_reduces_to_scalar() -> None:
    """A vector increment with m = 1 reproduces the scalar control row exactly."""
    rng = np.random.default_rng(4)
    readout_phi = rng.standard_normal((30, 11, 4))
    scalar_increments = rng.standard_normal((30, 10))
    scalar = control_features(readout_phi, scalar_increments)
    vector = control_features(readout_phi, scalar_increments[:, :, None])
    assert vector.shape == scalar.shape
    assert np.allclose(scalar, vector)


def test_assemble_control_features_handles_vector_increments() -> None:
    """The reservoir assembler accepts (n, N, m) shocks: shape and value agree."""
    paths = MODEL.simulate(200, 40, np.random.default_rng(5))
    features = asian_features(paths)
    reservoir = RandomizedSignatureReservoir.draw(3, 32, np.random.default_rng(6))
    # Fabricate a 2-direction increment from the scalar Brownian shock (algebra test).
    increments = np.stack(
        [paths.brownian_increments, np.roll(paths.brownian_increments, 1, axis=1)], axis=2
    )                                                       # (n, N, 2)
    phi = assemble_control_features(reservoir, features, paths.times, increments)
    assert phi.shape == (200, 2 * (32 + 2))
    states = reservoir.run(features)
    manual = control_features(readout_features(states, paths.times), increments)
    assert np.allclose(phi, manual)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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

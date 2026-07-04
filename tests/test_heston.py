"""Tests for the Heston simulator, its features, and the vector martingale control.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_heston.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    HestonModel,
    RandomizedSignatureReservoir,
    RidgeMartingaleControl,
    arithmetic_asian_call,
    assemble_control_features,
    evaluate_control,
    heston_features,
    make_heston_datasets,
    tune_ridge_lambda,
)

MODEL = HestonModel(s0=100.0, v0=0.04, r=0.03, kappa=1.5, theta=0.04, xi=0.30, rho=-0.70, T=1.0)


def test_simulator_shapes_and_initial_values() -> None:
    """Spot, variance and the 2-D Brownian increments have the documented shapes."""
    paths = MODEL.simulate(n_paths=1_000, n_steps=50, rng=np.random.default_rng(0))
    assert paths.spot.shape == (1_000, 51)
    assert paths.variance.shape == (1_000, 51)
    assert paths.brownian_increments.shape == (1_000, 50, 2)
    assert paths.times.shape == (51,)
    assert np.allclose(paths.spot[:, 0], MODEL.s0)
    assert np.allclose(paths.variance[:, 0], MODEL.v0)
    assert np.isclose(paths.dt, MODEL.T / 50)


def test_variance_stays_nonnegative() -> None:
    """Full truncation keeps every simulated variance value >= 0."""
    paths = MODEL.simulate(5_000, 50, np.random.default_rng(1))
    assert np.all(paths.variance >= 0.0)


def test_brownian_increments_correlation_matches_rho() -> None:
    """The realised correlation of (dW^S, dW^v) is approximately rho."""
    paths = MODEL.simulate(20_000, 50, np.random.default_rng(2))
    dW = paths.brownian_increments
    realised = np.corrcoef(dW[:, :, 0].ravel(), dW[:, :, 1].ravel())[0, 1]
    assert abs(realised - MODEL.rho) < 0.02


def test_discounted_spot_is_martingale() -> None:
    """E[e^{-r t_k} S_k] = S_0 within Monte Carlo noise (log-Euler keeps this exact)."""
    paths = MODEL.simulate(100_000, 50, np.random.default_rng(3))
    discount = np.exp(-MODEL.r * paths.times)
    discounted_mean = (discount * paths.spot).mean(axis=0)
    assert np.max(np.abs(discounted_mean - MODEL.s0)) < 0.5


def test_reproducibility() -> None:
    """The same seed reproduces identical spot, variance and increments."""
    a = MODEL.simulate(500, 50, np.random.default_rng(7))
    b = MODEL.simulate(500, 50, np.random.default_rng(7))
    assert np.array_equal(a.spot, b.spot)
    assert np.array_equal(a.variance, b.variance)
    assert np.array_equal(a.brownian_increments, b.brownian_increments)


def test_features_shapes_and_start_at_zero() -> None:
    """Default Heston features are d = 4 and every channel starts at 0."""
    paths = MODEL.simulate(100, 50, np.random.default_rng(4))
    feats = heston_features(paths)
    assert feats.values.shape == (100, 51, 4)
    assert feats.names == ("time", "log_price", "variance", "running_average")
    assert np.allclose(feats.values[:, 0, :], 0.0)
    rich = heston_features(paths, include_running_max=True, include_running_min=True)
    assert rich.values.shape == (100, 51, 6)


def test_make_heston_datasets_sizes_and_independence() -> None:
    """Datasets have the configured sizes and the splits are independent."""
    data = make_heston_datasets(MODEL, {"train": 2_000, "val": 1_000, "test": 3_000}, n_steps=50)
    assert data["train"].spot.shape == (2_000, 51)
    assert data["val"].variance.shape == (1_000, 51)
    assert data["test"].brownian_increments.shape == (3_000, 50, 2)
    # Independent sub-streams: the first increments of train and val differ.
    assert data["train"].brownian_increments[0, 0, 0] != data["val"].brownian_increments[0, 0, 0]


def test_invalid_parameters_raise() -> None:
    """Bad Heston parameters are rejected early."""
    for bad in (dict(s0=0.0), dict(v0=-0.1), dict(kappa=-1.0), dict(rho=1.5), dict(T=-1.0)):
        try:
            HestonModel(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_end_to_end_vector_control() -> None:
    """A small Heston pipeline runs, doubles the row width (m = 2), stays finite/unbiased."""
    data = make_heston_datasets(
        MODEL, {"train": 6_000, "val": 3_000, "test": 8_000}, n_steps=50, seed=11
    )
    feats = {name: heston_features(paths) for name, paths in data.items()}
    payoff = {name: arithmetic_asian_call(paths, strike=100.0, rate=MODEL.r)
              for name, paths in data.items()}

    dim = 48
    reservoir = RandomizedSignatureReservoir.draw(
        feats["train"].values.shape[2], dim, np.random.default_rng(99)
    )
    phi = {
        name: assemble_control_features(
            reservoir, feats[name], data[name].times, data[name].brownian_increments
        )
        for name in data
    }
    assert phi["train"].shape[1] == 2 * (dim + 2)      # vector m = 2 -> one block per direction

    lam, _ = tune_ridge_lambda(
        phi["train"], payoff["train"], phi["val"], payoff["val"], (0.0, 10.0, 1_000.0)
    )
    control = RidgeMartingaleControl.fit(phi["train"], payoff["train"], lam)
    report = evaluate_control(control, phi["test"], payoff["test"])

    assert np.isfinite(report.price_mc) and np.isfinite(report.price_cv)
    assert np.isfinite(report.variance_reduction)
    # The vector control is unbiased and reduces variance out of sample.
    assert abs(report.control_mean) < 5.0 * report.se_mc
    assert report.variance_reduction > 1.0


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

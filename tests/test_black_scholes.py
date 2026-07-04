"""Sanity-check tests for the Black-Scholes simulator and payoffs.

These run with pytest (``pytest tests``) *or* as a plain script
(``python tests/test_black_scholes.py``) so no extra dependency is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    BlackScholesModel,
    ExperimentConfig,
    arithmetic_asian_call,
    black_scholes_call_price,
    european_call,
    make_datasets,
)

MODEL = BlackScholesModel(s0=100.0, r=0.03, sigma=0.20, T=1.0)


def test_shapes_and_initial_value() -> None:
    """Paths and innovations have the documented shapes; S_0 is exact."""
    paths = MODEL.simulate(n_paths=1_000, n_steps=50, rng=np.random.default_rng(0))
    assert paths.spot.shape == (1_000, 51)
    assert paths.innovations.shape == (1_000, 50)
    assert paths.times.shape == (51,)
    assert np.allclose(paths.spot[:, 0], MODEL.s0)
    assert np.isclose(paths.dt, MODEL.T / 50)


def test_reproducibility() -> None:
    """The same seed reproduces identical paths; a different seed does not."""
    a = MODEL.simulate(2_000, 50, np.random.default_rng(42))
    b = MODEL.simulate(2_000, 50, np.random.default_rng(42))
    c = MODEL.simulate(2_000, 50, np.random.default_rng(7))
    assert np.array_equal(a.spot, b.spot)
    assert not np.array_equal(a.spot, c.spot)


def test_discounted_price_is_martingale() -> None:
    """E[e^{-r t_k} S_{t_k}] = S_0 for every k (within Monte Carlo noise)."""
    paths = MODEL.simulate(100_000, 50, np.random.default_rng(123))
    discount = np.exp(-MODEL.r * paths.times)
    discounted_mean = (discount * paths.spot).mean(axis=0)
    assert np.max(np.abs(discounted_mean - MODEL.s0)) < 0.25


def test_european_call_matches_closed_form() -> None:
    """Monte Carlo European-call price matches the analytic Black-Scholes price."""
    paths = MODEL.simulate(200_000, 50, np.random.default_rng(2024))
    payoff = european_call(paths, strike=100.0, rate=MODEL.r)
    mc_price = payoff.mean()
    analytic = black_scholes_call_price(
        s0=MODEL.s0, strike=100.0, rate=MODEL.r, sigma=MODEL.sigma, maturity=MODEL.T
    )
    assert abs(mc_price - analytic) < 0.10


def test_asian_payoff_is_nonnegative() -> None:
    """A call payoff can never be negative."""
    paths = MODEL.simulate(5_000, 50, np.random.default_rng(1))
    payoff = arithmetic_asian_call(paths, strike=100.0, rate=MODEL.r)
    assert np.all(payoff >= 0.0)


def test_make_datasets_sizes_and_independence() -> None:
    """Datasets have the configured sizes and the splits are independent."""
    config = ExperimentConfig(n_train=2_000, n_val=1_000, n_test=3_000, n_steps=50)
    data = make_datasets(config)
    assert data["train"].spot.shape == (2_000, 51)
    assert data["val"].spot.shape == (1_000, 51)
    assert data["test"].spot.shape == (3_000, 51)
    # Independent sub-streams: the first innovations of train and val differ.
    assert data["train"].innovations[0, 0] != data["val"].innovations[0, 0]


def test_invalid_parameters_raise() -> None:
    """Bad model parameters and simulation sizes are rejected early."""
    for bad in (dict(sigma=-0.1), dict(s0=0.0), dict(T=-1.0)):
        try:
            BlackScholesModel(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    try:
        MODEL.simulate(n_paths=0, n_steps=50, rng=np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_paths=0")


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

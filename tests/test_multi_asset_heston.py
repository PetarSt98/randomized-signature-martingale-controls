"""Tests for the three-asset Heston basket benchmark components."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    MultiAssetHestonModel,
    RandomizedSignatureReservoir,
    basket_arithmetic_asian_call,
    assemble_control_features,
    multi_asset_heston_features,
)

MODEL = MultiAssetHestonModel()


def test_multi_asset_heston_shapes_and_variance() -> None:
    paths = MODEL.simulate(n_paths=500, n_steps=12, rng=np.random.default_rng(0))
    assert paths.spot.shape == (500, 13, 3)
    assert paths.variance.shape == (500, 13, 3)
    assert paths.brownian_increments.shape == (500, 12, 6)
    assert paths.times.shape == (13,)
    assert np.allclose(paths.spot[:, 0, :], np.asarray(MODEL.s0))
    assert np.all(paths.variance >= 0.0)


def test_multi_asset_correlation_validation_rejects_invalid_matrix() -> None:
    bad = np.eye(6)
    bad[0, 1] = 0.5
    try:
        MultiAssetHestonModel(correlation=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-symmetric correlation")

    bad_psd = np.eye(6)
    bad_psd[0, 1] = bad_psd[1, 0] = 1.2
    try:
        MultiAssetHestonModel(correlation=bad_psd)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-PSD correlation")


def test_basket_payoff_and_features() -> None:
    paths = MODEL.simulate(300, 10, np.random.default_rng(1))
    payoff = basket_arithmetic_asian_call(paths, strike=100.0, rate=MODEL.r)
    assert payoff.shape == (300,)
    assert np.all(payoff >= 0.0)

    features = multi_asset_heston_features(paths)
    assert features.values.shape == (300, 11, 8)
    assert features.n_channels == 8
    assert features.names == (
        "time",
        "log_price_1",
        "log_price_2",
        "log_price_3",
        "variance_1",
        "variance_2",
        "variance_3",
        "running_basket_average",
    )
    assert np.allclose(features.values[:, 0, :], 0.0)


def test_multi_asset_control_assembly_handles_six_increments() -> None:
    paths = MODEL.simulate(120, 8, np.random.default_rng(2))
    features = multi_asset_heston_features(paths)
    reservoir = RandomizedSignatureReservoir.draw(
        n_channels=features.n_channels, dim=10, rng=np.random.default_rng(3)
    )
    phi = assemble_control_features(reservoir, features, paths.times, paths.brownian_increments)
    assert phi.shape == (120, 6 * (10 + 2))


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

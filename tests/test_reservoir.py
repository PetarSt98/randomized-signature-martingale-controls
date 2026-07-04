"""Tests for the Asian-aware features and the randomized-signature reservoir.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_reservoir.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    BlackScholesModel,
    RandomizedSignatureReservoir,
    asian_features,
    readout_features,
)
from rsmc.features import PathFeatures  # noqa: E402

MODEL = BlackScholesModel(s0=100.0, r=0.03, sigma=0.20, T=1.0)


def _paths(n_paths: int = 2_000, n_steps: int = 50, seed: int = 0):
    return MODEL.simulate(n_paths, n_steps, np.random.default_rng(seed))


def test_features_have_three_channels_starting_at_zero() -> None:
    """Default features are (time, log-price, running average) and start at 0."""
    features = asian_features(_paths())
    assert features.n_channels == 3
    assert features.names == ("time", "log_price", "running_average")
    assert features.values.shape == (2_000, 51, 3)
    assert features.increments.shape == (2_000, 50, 3)
    assert np.allclose(features.values[:, 0, :], 0.0)


def test_feature_ablation() -> None:
    """Disabling channels reduces d; at least one must remain."""
    features = asian_features(_paths(), include_running_average=False)
    assert features.names == ("time", "log_price")
    try:
        asian_features(
            _paths(),
            include_time=False,
            include_log_price=False,
            include_running_average=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when all channels are disabled")


def test_running_max_min_channels() -> None:
    """Optional max/min channels give d = 5, start at 0, and bracket log-price."""
    paths = _paths()
    features = asian_features(paths, include_running_max=True, include_running_min=True)
    assert features.n_channels == 5
    assert features.names == (
        "time", "log_price", "running_average", "running_max", "running_min",
    )
    assert np.allclose(features.values[:, 0, :], 0.0)

    log_price = features.values[:, :, 1]
    running_max = features.values[:, :, 3]
    running_min = features.values[:, :, 4]
    # Predictable running statistics: monotone and bracketing the current value.
    assert np.all(np.diff(running_max, axis=1) >= -1e-12)      # non-decreasing
    assert np.all(np.diff(running_min, axis=1) <= 1e-12)       # non-increasing
    assert np.all(running_max + 1e-12 >= log_price)
    assert np.all(running_min - 1e-12 <= log_price)


def test_reservoir_shapes_and_zero_initial_state() -> None:
    """Reservoir returns (n, N+1, D) states with h_0 = 0."""
    features = asian_features(_paths())
    reservoir = RandomizedSignatureReservoir.draw(
        n_channels=3, dim=64, rng=np.random.default_rng(1)
    )
    assert reservoir.dim == 64
    assert reservoir.n_channels == 3
    states = reservoir.run(features)
    assert states.shape == (2_000, 51, 64)
    assert np.allclose(states[:, 0, :], 0.0)


def test_reservoir_is_frozen_and_reproducible() -> None:
    """Same draw seed -> identical reservoir and states; different seed -> not."""
    features = asian_features(_paths())
    a = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(7))
    b = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(7))
    c = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(8))
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.run(features), b.run(features))
    assert not np.array_equal(a.weights, c.weights)


def test_reservoir_is_non_anticipative() -> None:
    """h_k must depend only on the path prefix x_0..x_k (no look-ahead)."""
    features = asian_features(_paths())
    reservoir = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(2))
    states = reservoir.run(features)

    k0 = 25
    rng = np.random.default_rng(99)
    corrupted = features.values.copy()
    corrupted[:, k0 + 1:, :] += rng.standard_normal(corrupted[:, k0 + 1:, :].shape)
    states_corrupted = reservoir.run(PathFeatures(corrupted, features.names))

    assert np.allclose(states[:, : k0 + 1, :], states_corrupted[:, : k0 + 1, :])
    assert not np.allclose(states[:, -1, :], states_corrupted[:, -1, :])


def test_readout_features_layout() -> None:
    """phi_k = [1, t_k, h_k] has the right shape and first two columns."""
    paths = _paths()
    features = asian_features(paths)
    reservoir = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(3))
    states = reservoir.run(features)
    phi = readout_features(states, paths.times)
    assert phi.shape == (2_000, 51, 66)
    assert np.allclose(phi[:, :, 0], 1.0)
    assert np.allclose(phi[:, :, 1], paths.times)
    assert np.allclose(phi[:, :, 2:], states)


def test_fixed_reservoir_control_is_mean_zero() -> None:
    """A fixed-beta reservoir control has mean zero (predictable x mean-zero dM)."""
    paths = _paths(n_paths=40_000, seed=5)
    features = asian_features(paths)
    reservoir = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(4))
    states = reservoir.run(features)
    phi = readout_features(states, paths.times)

    control_feature = np.einsum("nkq,nk->nq", phi[:, :-1, :], paths.innovations)
    beta = np.random.default_rng(11).standard_normal(phi.shape[-1])
    control = control_feature @ beta

    mean = control.mean()
    stderr = control.std(ddof=1) / np.sqrt(control.size)
    assert abs(mean) < 5.0 * stderr


def test_run_rejects_channel_mismatch() -> None:
    """A reservoir drawn for d channels rejects features with a different d."""
    features = asian_features(_paths(), include_running_average=False)  # d = 2
    reservoir = RandomizedSignatureReservoir.draw(3, 64, np.random.default_rng(0))
    try:
        reservoir.run(features)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on channel mismatch")


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

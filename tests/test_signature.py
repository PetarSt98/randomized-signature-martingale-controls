"""Tests for the finite truncated-signature benchmark features.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_signature.py``).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    BlackScholesModel,
    RidgeMartingaleControl,
    arithmetic_asian_call,
    asian_features,
    assemble_signature_control_features,
    running_signature,
    signature_dim,
    variance_reduction,
)

MODEL = BlackScholesModel(s0=100.0, r=0.03, sigma=0.20, T=1.0)
STRIKE = 100.0


def test_signature_dim_matches_table() -> None:
    """d = 3 dimensions, excluding and including the constant level 0."""
    assert signature_dim(3, 2) == 12
    assert signature_dim(3, 3) == 39
    assert signature_dim(3, 4) == 120
    assert signature_dim(3, 3, include_level0=True) == 40


def test_single_increment_is_the_exponential() -> None:
    """After one increment the signature equals exp(dx): [dx, dx(x)dx/2]."""
    a = np.array([0.3, -0.1, 0.2])
    increments = np.zeros((1, 2, 3))
    increments[0, 0] = a                         # only the first increment feeds s_1
    states = running_signature(increments, level=2)   # (1, 2, 12)

    assert np.allclose(states[0, 0], 0.0)        # s_0 is trivial
    level1 = states[0, 1, :3]
    level2 = states[0, 1, 3:]
    assert np.allclose(level1, a)
    assert np.allclose(level2, np.outer(a, a).reshape(-1) / 2)


def test_level_one_is_the_net_increment() -> None:
    """Level-1 of s_k is the running sum of increments (x_k - x_0)."""
    rng = np.random.default_rng(0)
    increments = rng.standard_normal((4, 6, 3)) * 0.1
    states = running_signature(increments, level=3)

    expected = np.concatenate(
        [np.zeros((4, 1, 3)), np.cumsum(increments, axis=1)[:, :-1, :]], axis=1
    )
    assert np.allclose(states[:, :, :3], expected)


def test_chen_two_step_matches_manual() -> None:
    """s_2 = exp(a)(x)exp(b): level 2 is a(x)a/2 + a(x)b + b(x)b/2."""
    a = np.array([0.2, -0.3, 0.1])
    b = np.array([-0.1, 0.4, 0.05])
    increments = np.zeros((1, 3, 3))
    increments[0, 0] = a
    increments[0, 1] = b
    states = running_signature(increments, level=2)
    s2 = states[0, 2]

    assert np.allclose(s2[:3], a + b)
    expected_level2 = (
        np.outer(a, a) / 2 + np.outer(a, b) + np.outer(b, b) / 2
    ).reshape(-1)
    assert np.allclose(s2[3:], expected_level2)


def test_assemble_shape() -> None:
    """Control rows have shape (n, 2 + q_L)."""
    paths = MODEL.simulate(500, 50, np.random.default_rng(1))
    features = asian_features(paths)
    phi = assemble_signature_control_features(
        features, paths.times, paths.brownian_increments, level=3
    )
    assert phi.shape == (500, 2 + signature_dim(3, 3))      # 2 + 39 = 41


@lru_cache(maxsize=None)
def _build(level=3, n_train=6_000, n_test=20_000, n_steps=50, seed=0):
    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)

    def rows(n_paths, rng_seed):
        paths = MODEL.simulate(n_paths, n_steps, np.random.default_rng(rng_seed))
        phi = assemble_signature_control_features(
            asian_features(paths), paths.times, paths.brownian_increments, level
        )
        g = arithmetic_asian_call(paths, strike=STRIKE, rate=MODEL.r)
        return phi, g

    phi_train, g_train = rows(n_train, train_seed)
    phi_test, g_test = rows(n_test, test_seed)
    return phi_train, g_train, phi_test, g_test


def test_signature_control_is_unbiased_and_reduces_variance() -> None:
    """The signature control is mean-zero out of sample and cuts variance."""
    phi_train, g_train, phi_test, g_test = _build(level=3)
    control = RidgeMartingaleControl.fit(phi_train, g_train, lam=10.0)

    c = control.control(phi_test)
    stderr = c.std(ddof=1) / np.sqrt(c.size)
    assert abs(c.mean()) < 5.0 * stderr                    # unbiased
    assert variance_reduction(control, phi_test, g_test) > 1.5


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

"""Tests for the partial-information hedging pipeline (Phase II).

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_hedging.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc import (  # noqa: E402
    HestonModel,
    MultiAssetHestonModel,
    RandomizedSignatureReservoir,
    arithmetic_asian_call,
    assemble_control_features,
    discounted_traded_increments,
    evaluate_hedge,
    fit_hedge,
    hedging_features,
    make_heston_datasets,
    multi_asset_hedging_features,
    observe_variance,
)
from rsmc.hedging import assemble_linear_hedge_features  # noqa: E402

MODEL = HestonModel()
MULTI_MODEL = MultiAssetHestonModel()
LAMBDAS = (0.0, 1e-2, 1.0, 100.0)


def _splits(eta: float, n_steps: int = 40, sizes=(4_000, 2_000, 4_000), seed: int = 0):
    """Train/val/test Heston paths each carrying an independent noisy signal."""
    data = make_heston_datasets(
        MODEL, {"train": sizes[0], "val": sizes[1], "test": sizes[2]}, n_steps, seed=seed
    )
    noise_seeds = np.random.SeedSequence(seed + 1).spawn(3)
    return {
        name: (paths, observe_variance(paths.variance, eta, np.random.default_rng(ns)))
        for (name, paths), ns in zip(data.items(), noise_seeds)
    }


def _oracle_features(paths, signal):
    return hedging_features(paths.spot, paths.times, paths.variance - MODEL.v0, "variance")


def _raw_features(paths, signal):
    return hedging_features(paths.spot, paths.times, signal - MODEL.v0, "signal")


def _hedge(splits, feature_fn, assemble):
    """Fit a hedge on train/val and score it on test."""
    psi, payoff = {}, {}
    for name, (paths, signal) in splits.items():
        feats = feature_fn(paths, signal)
        increments = discounted_traded_increments(paths.spot, paths.times, MODEL.r)
        psi[name] = assemble(feats, paths.times, increments)
        payoff[name] = arithmetic_asian_call(paths, 100.0, MODEL.r)
    control, _ = fit_hedge(psi["train"], payoff["train"], psi["val"], payoff["val"], LAMBDAS)
    return evaluate_hedge(control, psi["test"], payoff["test"])


def test_discounted_traded_increments_are_martingale() -> None:
    """dS~ has the right shape and conditional mean zero (E[dS~_{k+1}] = 0 per step)."""
    paths = MODEL.simulate(50_000, 50, np.random.default_rng(0))
    increments = discounted_traded_increments(paths.spot, paths.times, MODEL.r)
    assert increments.shape == (50_000, 50)
    se = increments.std(axis=0) / np.sqrt(increments.shape[0])
    assert np.all(np.abs(increments.mean(axis=0)) < 6.0 * se)


def test_observe_variance_noise_stats() -> None:
    """The signal is the variance plus zero-mean noise of standard deviation eta."""
    paths = MODEL.simulate(20_000, 50, np.random.default_rng(0))
    signal = observe_variance(paths.variance, 0.02, np.random.default_rng(1))
    residual = signal - paths.variance
    assert abs(float(residual.mean())) < 5e-4
    assert abs(float(residual.std()) - 0.02) < 1e-3


def test_hedging_features_shapes_and_start_at_zero() -> None:
    """Base features are d = 3; an extra centred vol channel makes d = 4, all starting at 0."""
    paths = MODEL.simulate(100, 40, np.random.default_rng(0))
    base = hedging_features(paths.spot, paths.times)
    assert base.values.shape == (100, 41, 3)
    assert base.names == ("time", "log_price", "running_average")
    assert np.allclose(base.values[:, 0, :], 0.0)

    with_vol = hedging_features(paths.spot, paths.times, paths.variance - MODEL.v0, "variance")
    assert with_vol.values.shape == (100, 41, 4)
    assert np.allclose(with_vol.values[:, 0, :], 0.0)


def test_linear_assembler_width_is_d_plus_2() -> None:
    """A linear readout collapses to q = d + 2 (the [1, t_k, x_k] block)."""
    paths = MODEL.simulate(200, 40, np.random.default_rng(0))
    feats = hedging_features(paths.spot, paths.times, paths.variance - MODEL.v0, "variance")
    increments = discounted_traded_increments(paths.spot, paths.times, MODEL.r)
    psi = assemble_linear_hedge_features(feats, paths.times, increments)
    assert psi.shape == (200, feats.n_channels + 2)


def test_multi_asset_features_and_traded_increments() -> None:
    """The d=8 extension uses three traded stock increments, not six Brownian shocks."""
    paths = MULTI_MODEL.simulate(120, 8, np.random.default_rng(9))
    signal = observe_variance(paths.variance, 0.02, np.random.default_rng(10))
    increments = discounted_traded_increments(paths.spot, paths.times, MULTI_MODEL.r)
    assert increments.shape == (120, 8, 3)

    price_features = multi_asset_hedging_features(paths.spot, paths.times)
    assert price_features.values.shape == (120, 9, 5)

    observed_features = multi_asset_hedging_features(
        paths.spot,
        paths.times,
        signal - np.asarray(MULTI_MODEL.v0)[None, None, :],
        "signal",
    )
    assert observed_features.values.shape == (120, 9, 8)
    assert observed_features.names == (
        "time",
        "log_price_1",
        "log_price_2",
        "log_price_3",
        "signal_1",
        "signal_2",
        "signal_3",
        "running_basket_average",
    )
    psi = assemble_linear_hedge_features(observed_features, paths.times, increments)
    assert psi.shape == (120, 3 * (8 + 2))


def test_oracle_hedge_reduces_variance_and_breaks_even() -> None:
    """The full-information hedge cuts payoff variance and starts from ~ the fair price."""
    report = _hedge(_splits(eta=0.02), _oracle_features, assemble_linear_hedge_features)
    assert report.variance_reduction > 2.0
    assert abs(report.mean_error) < 5.0 * report.mean_error_se
    assert abs(report.capital - report.price) < 0.10 * report.payoff_std


def test_oracle_at_least_as_good_as_raw_signal() -> None:
    """Full information cannot hedge worse than the noisy observed signal."""
    splits = _splits(eta=0.02)
    oracle = _hedge(splits, _oracle_features, assemble_linear_hedge_features)
    raw = _hedge(splits, _raw_features, assemble_linear_hedge_features)
    assert oracle.variance_reduction >= raw.variance_reduction


def test_perfect_signal_matches_oracle() -> None:
    """With eta = 0 the signal equals the variance, so raw and oracle hedges coincide."""
    splits = _splits(eta=0.0)
    oracle = _hedge(splits, _oracle_features, assemble_linear_hedge_features)
    raw = _hedge(splits, _raw_features, assemble_linear_hedge_features)
    assert abs(oracle.variance_reduction - raw.variance_reduction) < 1e-6


def test_reservoir_hedge_runs_and_reduces_variance() -> None:
    """A frozen reservoir of the observed (S, Y) path yields a finite, useful hedge."""
    splits = _splits(eta=0.02)
    n_channels = _raw_features(*splits["train"]).n_channels
    reservoir = RandomizedSignatureReservoir.draw(n_channels, 64, np.random.default_rng(7))
    assemble = lambda feats, times, inc: assemble_control_features(reservoir, feats, times, inc)
    report = _hedge(splits, _raw_features, assemble)
    assert np.isfinite(report.variance_reduction)
    assert report.variance_reduction > 2.0


def test_hedging_demo_smoke_run() -> None:
    """The demo keeps the same baseline / reservoir / signature orchestration on tiny data."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "hedging_demo.py"
    spec = importlib.util.spec_from_file_location("hedging_demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows = module.run_demo(
        sizes={"train": 40, "val": 20, "test": 30},
        n_steps=3,
        reservoir_dims=(4,),
        signature_levels=(2,),
        lambdas=(1.0,),
        print_output=False,
    )
    assert len(rows) == 12
    assert {row["label"] for row in rows} == {
        "price-only linear",
        "observed signal linear",
        "oracle true V linear",
        "oracle reservoir D=4",
        "reservoir D=4",
        "signature L=2",
    }
    assert sorted({row["d"] for row in rows}) == [3, 4, 5, 8]
    assert {row["m"] for row in rows} == {1, 3}
    assert {row["n_train"] for row in rows} == {40}
    assert {row["n_val"] for row in rows} == {20}
    assert {row["n_eval"] for row in rows} == {30}


def test_hedging_demo_eta_sweep_smoke_run() -> None:
    """The eta sweep runs on tiny data and keeps validation-selected summaries available."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "hedging_demo.py"
    spec = importlib.util.spec_from_file_location("hedging_demo_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows = module.run_eta_sweep(
        etas=(0.0, 0.02),
        sizes={"train": 30, "val": 15, "test": 20},
        n_steps=3,
        reservoir_dims=(4,),
        signature_levels=(2,),
        lambdas=(1.0,),
        print_output=False,
    )
    assert len(rows) == 24
    assert sorted({row["eta"] for row in rows}) == [0.0, 0.02]
    assert {row["m"] for row in rows} == {1, 3}
    assert all("validation_mse" in row for row in rows)


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

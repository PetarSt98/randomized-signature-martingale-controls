"""Exact transition and information checks for the optional two-factor task."""

import numpy as np
import pytest

from rsmc import transfer_tasks
from rsmc.two_factor_heston import (
    TwoFactorHestonModel, TwoFactorHestonPaths, make_two_factor_heston_datasets,
)


def test_supplied_innovations_use_current_variances_and_separate_projections():
    model = TwoFactorHestonModel()
    log_spot = np.log([100., 97.])
    variance = np.array([[.04, .01], [0., 0.]])
    normals = np.array([[[.5, -12.], [-.4, .25]], [[3., 4.], [5., 6.]]])
    dt = .02
    next_log, next_variance, increments = model.step(log_spot, variance, dt, normals)
    d_b = np.sqrt(dt) * normals[..., 0]
    d_w = np.sqrt(dt) * (-.7 * normals[..., 0] + np.sqrt(.51) * normals[..., 1])
    expected_log = log_spot + (.03 - .5 * variance.sum(axis=1)) * dt
    expected_log += (np.sqrt(variance) * d_b).sum(axis=1)
    expected_variance = np.maximum(
        variance + np.array([.5, 5.]) * (.02 - variance) * dt
        + .3 * np.sqrt(variance) * d_w, 0.)
    np.testing.assert_allclose(next_log, expected_log, rtol=0, atol=1e-15)
    np.testing.assert_allclose(next_variance, expected_variance, rtol=0, atol=1e-17)
    np.testing.assert_allclose(increments, np.stack([d_b, d_w], axis=-1).reshape(2, 4))
    assert next_variance[0, 0] == 0.  # Projection acts on this factor only.
    assert next_variance[0, 1] > 0.
    np.testing.assert_array_equal(next_variance[1], [.0002, .002])
    assert next_log[1] == log_spot[1] + model.r * dt

    # Orthogonal variance shocks change next variance but not this stock step.
    changed = normals.copy()
    changed[..., 1] += 20.
    altered_log, altered_variance, _ = model.step(log_spot, variance, dt, changed)
    np.testing.assert_array_equal(next_log, altered_log)
    assert not np.array_equal(next_variance, altered_variance)


def test_innovation_map_has_only_declared_within_pair_correlations():
    # Images of the four basis vectors identify the Gaussian linear map exactly.
    model = TwoFactorHestonModel(rho=(-.7, .25))
    _, _, transform = model.step(np.zeros(4), np.zeros((4, 2)), 1.,
                                 np.eye(4).reshape(4, 2, 2))
    expected = np.array([[1., -.7, 0., 0.], [-.7, 1., 0., 0.],
                         [0., 0., 1., .25], [0., 0., .25, 1.]])
    np.testing.assert_allclose(transform.T @ transform, expected, atol=1e-15)


def test_simulator_is_reproducible_and_future_innovations_do_not_change_past():
    model = TwoFactorHestonModel()
    first = model.simulate(6, 7, np.random.default_rng(103))
    repeat = model.simulate(6, 7, np.random.default_rng(103))
    assert first.spot.shape == (6, 8)
    assert first.variance.shape == (6, 8, 2)
    assert first.brownian_increments.shape == (6, 7, 4)
    assert first.n_paths == 6 and first.n_steps == 7 and first.dt == 1 / 7
    for name in ("spot", "variance", "brownian_increments", "times"):
        np.testing.assert_array_equal(getattr(first, name), getattr(repeat, name))
    assert np.all(first.variance >= 0) and np.all(first.spot > 0)

    class SuppliedNormals:
        def __init__(self, values):
            self.values = values

        def standard_normal(self, shape):
            assert shape == self.values.shape
            return self.values.copy()

    normals = np.random.default_rng(104).standard_normal((6, 7, 2, 2))
    changed = normals.copy()
    changed[:, 3:] += 4.
    before = model.simulate(6, 7, SuppliedNormals(normals))
    after = model.simulate(6, 7, SuppliedNormals(changed))
    np.testing.assert_array_equal(before.spot[:, :4], after.spot[:, :4])
    np.testing.assert_array_equal(before.variance[:, :4], after.variance[:, :4])
    assert not np.array_equal(before.spot[:, 4:], after.spot[:, 4:])


def test_dataset_child_streams_keep_training_and_validation_fixed_before_test():
    model = TwoFactorHestonModel()
    earlier = make_two_factor_heston_datasets(model, {"train": 8, "val": 5}, 6, 105)
    later = make_two_factor_heston_datasets(model, {"train": 8, "val": 5, "test": 7}, 6, 105)
    for name in earlier:
        np.testing.assert_array_equal(earlier[name].spot, later[name].spot)
        np.testing.assert_array_equal(earlier[name].variance, later[name].variance)
    assert not np.array_equal(later["train"].brownian_increments[0],
                              later["val"].brownian_increments[0])


def test_adapter_exposes_total_variance_only_and_keeps_payoff_and_premium(monkeypatch):
    times = np.linspace(0., 1., 4)
    spot = np.array([[100., 110., 120., 130.], [100., 80., 95., 105.],
                     [100., 102., 108., 115.]])
    variance = np.broadcast_to([.01, .03], (3, 4, 2)).copy()
    paths = TwoFactorHestonPaths(times, spot, variance, np.zeros((3, 3, 4)))
    monkeypatch.setattr(transfer_tasks, "make_two_factor_heston_datasets",
                        lambda *args: {"train": paths, "val": paths})
    data, grid, rate, premium, metadata = transfer_tasks.make_transfer_dataset(
        "two_factor_heston", {"train": 3, "val": 3}, 3, 106)
    expected_payoff = np.exp(-.03) * np.maximum(spot[:, 1:].mean(axis=1) - 100., 0.)
    np.testing.assert_array_equal(data["train"]["payoff"], expected_payoff)
    assert premium == float(np.float32(expected_payoff.mean()))
    assert rate == .03
    np.testing.assert_array_equal(grid, times)
    assert set(data["train"]) == {"features", "spot", "payoff"}
    features = data["train"]["features"]
    assert features.names == ("time", "log_price", "running_average", "total_variance")
    np.testing.assert_allclose(features.values[..., 1], np.log(spot / 100.))
    np.testing.assert_allclose(features.values[..., 2],
                               np.log(np.cumsum(spot, axis=1) / np.arange(1, 5) / 100.))
    np.testing.assert_array_equal(features.values[..., 3], np.ones((3, 4)))
    assert metadata["payoff_includes_initial"] is False
    assert metadata["running_feature_includes_initial"] is True

    # Same declared observations, different latent decomposition and innovations.
    paths.variance[:] = [.03, .01]
    paths.brownian_increments[:] = 123.
    other, _, _, _, other_meta = transfer_tasks.make_transfer_dataset(
        "two_factor_heston", {"train": 3, "val": 3}, 3, 106)
    np.testing.assert_array_equal(features.values, other["train"]["features"].values)
    assert metadata["dataset_features_sha256"] == other_meta["dataset_features_sha256"]


@pytest.mark.parametrize("kwargs", [dict(v0=(.02,)), dict(kappa=(.5, -5.)),
                                    dict(rho=(-.7, 1.1)), dict(T=float("nan"))])
def test_invalid_model_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        TwoFactorHestonModel(**kwargs)

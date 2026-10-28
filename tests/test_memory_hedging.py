"""Information, fitting and cash-account checks for the memory comparison."""

import numpy as np
import pytest

from rsmc.features import PathFeatures
from rsmc.hedging import (
    FrozenTanhFeatures,
    assemble_linear_hedge_features,
    assemble_reservoir_hedge_features,
    assemble_static_hedge_features,
    discounted_traded_increments,
    evaluate_hedge,
    fit_fixed_premium_hedge,
    hedging_features,
    multi_asset_hedging_features,
)
from rsmc.reservoir import RandomizedSignatureReservoir, readout_features


def test_static_features_depend_only_on_declared_current_summaries():
    # Different order of past prices, same current price and accrued average.
    spot = np.array([[100., 90., 110., 105., 108.],
                     [100., 110., 90., 105., 108.]])
    times = np.linspace(0., 1., spot.shape[1])
    summaries = hedging_features(spot, times)
    static = FrozenTanhFeatures.draw(3, 8, np.random.default_rng(201))
    before = summaries.values.copy()
    values = static.run(summaries)
    assert not np.array_equal(summaries.values[0, 1], summaries.values[1, 1])
    np.testing.assert_array_equal(summaries.values[0, 3], summaries.values[1, 3])
    np.testing.assert_array_equal(values[0, 3:], values[1, 3:])

    # This also holds without making any assumption about admissible earlier states.
    changed = summaries.values.copy()
    changed[:, :3] += 10.
    altered = static.run(PathFeatures(changed, summaries.names))
    np.testing.assert_array_equal(values[:, 3:], altered[:, 3:])
    np.testing.assert_array_equal(summaries.values, before)


def test_static_draws_are_reproducible_and_frozen_through_readout_fit():
    times = np.linspace(0., 1., 5)
    spot = np.array([[100., 98., 101., 103., 106.],
                     [100., 102., 99., 101., 104.],
                     [100., 103., 105., 106., 108.],
                     [100., 97., 95., 99., 101.]])
    summaries = hedging_features(spot, times)
    static = FrozenTanhFeatures.draw(3, 5, np.random.default_rng(202), .5, 1.)
    repeat = FrozenTanhFeatures.draw(3, 5, np.random.default_rng(202), .5, 1.)
    before = {k: v.copy() for k, v in vars(static).items() if isinstance(v, np.ndarray)}
    np.testing.assert_array_equal(static.run(summaries), repeat.run(summaries))
    increments = discounted_traded_increments(spot, times, .03)
    psi = assemble_static_hedge_features(static, summaries, times, increments)
    fitted = fit_fixed_premium_hedge(psi, np.array([3., 0., 5., 1.]), 2.25, .1)
    evaluate_hedge(fitted, psi, np.array([3., 0., 5., 1.]))
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(static, name), value)


def test_nonlinear_gain_rows_have_identical_current_summary_passthrough():
    times = np.linspace(0., 1., 5)
    spot = np.array([[100., 99., 104., 103., 106.],
                     [100., 102., 98., 100., 101.]])
    summaries = hedging_features(spot, times)
    increments = discounted_traded_increments(spot, times, .03)
    static = FrozenTanhFeatures.draw(3, 7, np.random.default_rng(203))
    reservoir = RandomizedSignatureReservoir.draw(3, 7, np.random.default_rng(204))
    affine = assemble_linear_hedge_features(summaries, times, increments)
    static_rows = assemble_static_hedge_features(static, summaries, times, increments)
    reservoir_rows = assemble_reservoir_hedge_features(reservoir, summaries, times, increments)
    assert static_rows.shape == reservoir_rows.shape == (2, 11)
    np.testing.assert_array_equal(static_rows[:, :4], affine)
    np.testing.assert_array_equal(reservoir_rows[:, :4], affine)


def test_reservoir_holdings_cannot_use_future_prices():
    times = np.linspace(0., 1., 5)
    spot = np.array([[100., 99., 104., 103., 106.]])
    changed_spot = spot.copy()
    changed_spot[:, 3:] *= 4.
    original = hedging_features(spot, times)
    changed = hedging_features(changed_spot, times)
    reservoir = RandomizedSignatureReservoir.draw(3, 7, np.random.default_rng(205))
    beta = np.linspace(-1., 1., 11)

    def holdings(features):
        phi = readout_features(
            np.concatenate([features.values[:, :, 1:], reservoir.run(features)], axis=2),
            times,
        )
        return phi[:, :-1] @ beta

    np.testing.assert_array_equal(holdings(original)[:, :3], holdings(changed)[:, :3])
    assert not np.array_equal(holdings(original)[:, 3:], holdings(changed)[:, 3:])


def test_static_ridge_uses_training_gain_scales_on_shifted_validation_data():
    rng = np.random.default_rng(206)
    times = np.linspace(0., 1., 6)
    train_spot = 100. * np.exp(np.cumsum(rng.normal(0., .03, (30, 6)), axis=1))
    train_spot[:, 0] = 100.
    val_spot = 100. * np.exp(np.cumsum(rng.normal(.03, .25, (12, 6)), axis=1))
    val_spot[:, 0] = 100.
    static = FrozenTanhFeatures.draw(3, 4, np.random.default_rng(207))
    train = assemble_static_hedge_features(
        static, hedging_features(train_spot, times), times,
        discounted_traded_increments(train_spot, times, .03),
    )
    val = assemble_static_hedge_features(
        static, hedging_features(val_spot, times), times,
        discounted_traded_increments(val_spot, times, .03),
    )
    payoff = np.maximum(train_spot[:, 1:].mean(axis=1) - 100., 0.) * np.exp(-.03)
    val_payoff = np.maximum(val_spot[:, 1:].mean(axis=1) - 100., 0.) * np.exp(-.03)
    premium = float(np.float32(payoff.mean()))
    lam = 10.
    fitted = fit_fixed_premium_hedge(train, payoff, premium, lam)
    scale = np.sqrt(np.mean(train ** 2, axis=0))
    scale[scale < 1e-12] = 1.
    gamma, *_ = np.linalg.lstsq(
        np.vstack([train / scale, np.sqrt(lam) * np.eye(train.shape[1])]),
        np.r_[payoff - premium, np.zeros(train.shape[1])], rcond=None,
    )
    expected_errors = val_payoff - premium - (val / scale) @ gamma
    report = evaluate_hedge(fitted, val, val_payoff)
    np.testing.assert_allclose(report.errors, expected_errors, rtol=1e-10, atol=1e-10)
    assert report.capital == premium
    assert report.error_mse == pytest.approx(np.mean(expected_errors ** 2))
    assert report.rmse == pytest.approx(np.sqrt(np.mean(expected_errors ** 2)))
    assert abs(report.mean_error) > .1  # The validation residual was not recentered.
    contaminated = fit_fixed_premium_hedge(
        np.r_[train, val], np.r_[payoff, val_payoff], premium, lam,
    )
    assert not np.allclose(fitted.beta, contaminated.beta)


@pytest.mark.parametrize("n_assets", [1, 3])
def test_static_gain_rows_equal_nominal_self_financing_account_with_liquidation(n_assets):
    rng = np.random.default_rng(208)
    times = np.linspace(0., 1., 6)
    stock = 100. * np.exp(np.cumsum(rng.normal(0., .07, (3, 6, n_assets)), axis=1))
    stock[:, 0] = 100.
    spot = stock[:, :, 0] if n_assets == 1 else stock
    summaries = (hedging_features(spot, times) if n_assets == 1
                 else multi_asset_hedging_features(spot, times))
    static = FrozenTanhFeatures.draw(summaries.n_channels, 5, np.random.default_rng(209))
    phi = readout_features(np.concatenate(
        [summaries.values[:, :, 1:], static.run(summaries)], axis=2), times)
    beta = np.linspace(-1.2, .8, n_assets * phi.shape[2])
    theta = phi[:, :-1] @ beta.reshape(n_assets, -1).T
    premium, rate = 4.25, .03
    # Independent nominal account: pay for initial shares, accrue cash interest,
    # pay for each rebalance, then sell all remaining shares at maturity.
    cash = np.full(spot.shape[0], premium) - (theta[:, 0] * stock[:, 0]).sum(axis=1)
    for k in range(1, len(times)):
        cash *= np.exp(rate * (times[k] - times[k - 1]))
        next_position = theta[:, k] if k < theta.shape[1] else np.zeros_like(theta[:, k - 1])
        cash -= ((next_position - theta[:, k - 1]) * stock[:, k]).sum(axis=1)
    psi = assemble_static_hedge_features(
        static, summaries, times, discounted_traded_increments(spot, times, rate),
    )
    np.testing.assert_allclose(cash * np.exp(-rate * times[-1]),
                               premium + psi @ beta, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("d", [3, 5])
def test_window_zero_reproduces_original_static_draw_and_features(d):
    rng = np.random.default_rng(301)
    original_weights = rng.standard_normal((64, d)) * (.5 / np.sqrt(d))
    original_biases = rng.standard_normal(64)
    summaries = PathFeatures(np.arange(4*6*d).reshape(4, 6, d) / 100.,
                             tuple(f"channel_{j}" for j in range(d)))
    window_zero = FrozenTanhFeatures.draw(d, 64, np.random.default_rng(301), .5, 1., window=0)
    np.testing.assert_array_equal(window_zero.weights, original_weights)
    np.testing.assert_array_equal(window_zero.biases, original_biases)
    np.testing.assert_array_equal(
        window_zero.run(summaries),
        np.tanh(summaries.values @ original_weights.T + original_biases),
    )


def test_window_order_and_inception_padding_use_whole_nonzero_initial_vector():
    summaries = PathFeatures(np.array([[[.1, .2], [.3, .4], [.5, .6]]]), ("a", "b"))
    # Identity weights make every lag coordinate separately observable.
    model = FrozenTanhFeatures(np.eye(6), np.zeros(6), window=2)
    expected = np.array([[[.1, .2, .1, .2, .1, .2],
                          [.3, .4, .1, .2, .1, .2],
                          [.5, .6, .3, .4, .1, .2]]])
    np.testing.assert_array_equal(model.run(summaries), np.tanh(expected))


@pytest.mark.parametrize("window", [0, 5, 20])
def test_window_features_and_positions_are_causal_and_have_no_hidden_history(window):
    from rsmc.cost_hedging import hedge_readout_features

    times = np.linspace(0., 1., 31)
    spot = (100. + np.arange(31) + np.sin(np.arange(31)))[None, :]
    original = hedging_features(spot, times)
    changed_spot = spot.copy()
    changed_spot[:, 16:] *= 1.5
    future_changed = hedging_features(changed_spot, times)
    model = FrozenTanhFeatures.draw(3, 64, np.random.default_rng(302), window=window)
    beta = np.linspace(-.2, .3, 68)
    original_phi = hedge_readout_features(original, times, model)
    changed_phi = hedge_readout_features(future_changed, times, model)
    np.testing.assert_array_equal(original_phi[:, :16], changed_phi[:, :16])
    np.testing.assert_array_equal((original_phi[:, :-1] @ beta)[:, :16],
                                  (changed_phi[:, :-1] @ beta)[:, :16])
    assert not np.array_equal(original_phi[:, 16:], changed_phi[:, 16:])

    # Hold the declared window summaries fixed; arbitrary older feature history
    # must not persist in this map, including across repeated calls.
    older_changed = original.values.copy()
    older_changed[:, :25-window] += .7
    altered = model.run(PathFeatures(older_changed, original.names))
    np.testing.assert_array_equal(model.run(original)[:, 25], altered[:, 25])


@pytest.mark.parametrize("window", [0, 5, 20])
@pytest.mark.parametrize("n_assets", [1, 3])
def test_window_dimensions_passthrough_and_fitted_gain_accounting(window, n_assets):
    from rsmc.cost_hedging import hedge_readout_features

    rng = np.random.default_rng(303)
    times = np.linspace(0., 1., 8)
    stock = 100. * np.exp(np.cumsum(rng.normal(0., .03, (260, 8, n_assets)), axis=1))
    stock[:, 0] = 100.
    spot = stock[:, :, 0] if n_assets == 1 else stock
    summaries = (hedging_features(spot, times) if n_assets == 1
                 else multi_asset_hedging_features(spot, times))
    d = summaries.n_channels
    model = FrozenTanhFeatures.draw(d, 64, np.random.default_rng(304), .5, 1., window=window)
    input_dim = d * (window + 1)
    expected_rng = np.random.default_rng(304)
    np.testing.assert_array_equal(
        model.weights, expected_rng.standard_normal((64, input_dim)) * (.5 / np.sqrt(input_dim)),
    )
    np.testing.assert_array_equal(model.biases, expected_rng.standard_normal(64))
    assert model.n_channels == d
    phi = hedge_readout_features(summaries, times, model)
    q = 64 + d + 1
    assert phi.shape == (260, 8, q)
    np.testing.assert_array_equal(phi[:, :, :d+1], hedge_readout_features(summaries, times))
    increments = discounted_traded_increments(spot, times, .03)
    psi = assemble_static_hedge_features(model, summaries, times, increments)
    assert psi.shape == (260, 68 if n_assets == 1 else 210)
    payoff = np.exp(-.03) * np.maximum(stock[:, 1:].mean(axis=(1, 2)) - 100., 0.)
    premium = float(np.float32(payoff[:240].mean()))
    fitted = fit_fixed_premium_hedge(psi[:240], payoff[:240], premium, .1)
    positions = phi[:, :-1] @ fitted.beta.reshape(n_assets, q).T
    discounted_stock = stock * np.exp(-.03 * times)[None, :, None]
    direct_gain = (positions * np.diff(discounted_stock, axis=1)).sum(axis=(1, 2))
    np.testing.assert_allclose(psi @ fitted.beta, direct_gain, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        evaluate_hedge(fitted, psi[240:], payoff[240:]).errors,
        payoff[240:] - premium - direct_gain[240:], rtol=1e-10, atol=1e-10,
    )

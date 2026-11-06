"""Volterra simulator, leaky features, risk metrics and basket cost-Transformer checks."""
import numpy as np
import pytest

from rsmc import transfer_tasks
from rsmc.cost_hedging import CostSpec, cost_metrics, execution_cashflows
from rsmc.features import PathFeatures
from rsmc.hedging import FrozenTanhFeatures, LeakyTanhFeatures, hedging_features
from rsmc.volterra import BoundedVolterraModel, make_volterra_datasets


def test_kernel_weights_match_closed_form_and_markov_case_is_flat():
    model = BoundedVolterraModel(hurst=0.1)
    g = model.kernel_weights(50)
    dt = 1 / 50
    m = np.arange(1, 51)
    expected = np.sqrt(0.2) * dt ** (-0.4) / 0.6 * (m ** 0.6 - (m - 1) ** 0.6)
    assert g[0] == 0.0
    np.testing.assert_allclose(g[1:], expected)
    assert np.all(np.diff(g[1:]) < 0)  # power-law decay: history matters
    np.testing.assert_allclose(BoundedVolterraModel(hurst=0.5).kernel_weights(7)[1:], 1.0)
    # Cell averaging keeps the fraction 2H/(H+1/2)^2 of the continuous variance at t_1.
    assert g[1] ** 2 * dt / dt ** 0.2 == pytest.approx(0.2 / 0.36)


def test_driver_is_causal_weighted_sum_and_markov_driver_is_random_walk():
    model = BoundedVolterraModel(hurst=0.1)
    increments = np.array([[1.0, -2.0, 0.5], [0.3, 0.0, -1.0]])
    g = model.kernel_weights(3)
    x = model.driver(increments)
    assert x.shape == (2, 4) and np.all(x[:, 0] == 0.0)
    np.testing.assert_allclose(x[:, 1], g[1] * increments[:, 0])
    np.testing.assert_allclose(x[:, 2], g[2] * increments[:, 0] + g[1] * increments[:, 1])
    np.testing.assert_allclose(x[:, 3], g[3] * increments[:, 0] + g[2] * increments[:, 1] + g[1] * increments[:, 2])
    walk = BoundedVolterraModel(hurst=0.5).driver(increments)
    np.testing.assert_allclose(walk[:, 1:], np.cumsum(increments, axis=1))


def test_simulation_shapes_bounded_volatility_causality_and_discounted_martingale():
    model = BoundedVolterraModel()
    paths = model.simulate(20000, 10, np.random.default_rng(301))
    assert paths.spot.shape == (20000, 11) and paths.volatility.shape == (20000, 11)
    assert paths.brownian_increments.shape == (20000, 10, 2)
    assert np.all(paths.volatility > 0.05) and np.all(paths.volatility < 0.45)
    np.testing.assert_allclose(paths.volatility[:, 0], 0.25)
    discounted = paths.spot * np.exp(-model.r * paths.times)
    # E[S~_N] = S_0 within Monte Carlo error (bounded volatility => finite variance).
    se = discounted[:, -1].std(ddof=1) / np.sqrt(20000)
    assert abs(discounted[:, -1].mean() - 100.0) < 4 * se
    # Leverage: negative correlation between stock returns and volatility shocks.
    returns = np.diff(np.log(paths.spot), axis=1)
    assert np.corrcoef(returns[:, 0], paths.brownian_increments[:, 0, 1])[0, 1] < -0.4

    class SuppliedNormals:
        def __init__(self, values):
            self.values, self.calls = values, 0

        def standard_normal(self, shape):
            self.calls += 1
            return self.values[self.calls - 1].copy()

    normals = np.random.default_rng(302).standard_normal((2, 6, 8))
    changed = normals.copy()
    changed[:, :, 4:] += 3.0
    before = model.simulate(6, 8, SuppliedNormals(normals))
    after = model.simulate(6, 8, SuppliedNormals(changed))
    np.testing.assert_array_equal(before.spot[:, :5], after.spot[:, :5])
    np.testing.assert_array_equal(before.volatility[:, :5], after.volatility[:, :5])
    assert not np.array_equal(before.spot[:, 5:], after.spot[:, 5:])


def test_volterra_task_observes_scaled_volatility_and_keeps_fixed_premium():
    data, times, rate, premium, meta = transfer_tasks.make_transfer_dataset(
        "volterra", {"train": 30, "val": 10}, 5, 303)
    features = data["train"]["features"]
    assert features.names == ("time", "log_price", "running_average", "volatility")
    np.testing.assert_allclose(features.values[:, 0, 3], 1.0)
    assert premium == float(np.float32(data["train"]["payoff"].mean()))
    assert rate == 0.03 and meta["model"]["hurst"] == 0.1
    assert set(meta["dataset_features_sha256"]) == {"train", "val"}
    markov, *_ = transfer_tasks.make_transfer_dataset("volterra_markov", {"train": 30, "val": 10}, 5, 303)
    assert not np.array_equal(markov["train"]["spot"], data["train"]["spot"])
    earlier = make_volterra_datasets(BoundedVolterraModel(), {"train": 8, "val": 5}, 6, 304)
    later = make_volterra_datasets(BoundedVolterraModel(), {"train": 8, "val": 5, "test": 7}, 6, 304)
    for name in earlier:
        np.testing.assert_array_equal(earlier[name].spot, later[name].spot)


@pytest.mark.parametrize("kwargs", [dict(hurst=0.0), dict(hurst=0.7), dict(rho=1.2), dict(sigma_min=0.0)])
def test_invalid_volterra_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundedVolterraModel(**kwargs)


def test_leaky_features_are_causal_exponential_averages_of_static_features():
    spot = np.array([[100.0, 99.0, 104.0, 103.0, 106.0], [100.0, 102.0, 98.0, 100.0, 101.0]])
    times = np.linspace(0.0, 1.0, 5)
    summaries = hedging_features(spot, times)
    static = FrozenTanhFeatures.draw(3, 6, np.random.default_rng(305), 0.5, 1.0)
    leaky = LeakyTanhFeatures.draw(3, 6, np.random.default_rng(305), 0.5, 1.0, decay=0.8)
    np.testing.assert_array_equal(leaky.weights, static.weights)
    z = static.run(summaries)
    q = leaky.run(summaries)
    np.testing.assert_array_equal(q[:, 0], z[:, 0])
    expected = z[:, 0]
    for k in range(1, 5):
        expected = 0.8 * expected + 0.2 * z[:, k]
        np.testing.assert_allclose(q[:, k], expected)
    # Decay zero is the static map; the leaky state depends on earlier summaries.
    np.testing.assert_array_equal(LeakyTanhFeatures(static.weights, static.biases, decay=0.0).run(summaries), z)
    changed = summaries.values.copy()
    changed[:, 1, 1:] += 0.1
    altered = leaky.run(PathFeatures(changed, summaries.names))
    np.testing.assert_array_equal(altered[:, :1], q[:, :1])
    assert not np.allclose(altered[:, 2:], q[:, 2:])
    with pytest.raises(ValueError):
        LeakyTanhFeatures(static.weights, static.biases, decay=1.0)


def test_lbfgs_readout_improves_on_ridge_start_and_validates_with_exact_costs():
    pytest.importorskip("torch")
    from rsmc.cost_hedging import fit_cost_readout, gain_row_scale, hedge_readout_features, readout_positions
    from rsmc.hedging import fit_fixed_premium_hedge
    rng = np.random.default_rng(308)
    times = np.linspace(0.0, 1.0, 6)
    spot = 100.0 * np.exp(np.cumsum(np.column_stack([np.zeros(60), rng.normal(0, 0.03, (60, 5))]), axis=1))
    phi = hedge_readout_features(hedging_features(spot, times), times)
    psi, scale = gain_row_scale(phi, spot, times, 0.03)
    payoff = np.maximum(spot[:, 1:].mean(axis=1) - 100.0, 0.0)
    premium = float(payoff.mean())
    ridge = fit_fixed_premium_hedge(psi, payoff, premium, 0.1)
    costs = CostSpec(0.02, 0.0)
    fitted = fit_cost_readout(phi, spot, payoff, phi, spot, payoff, times, 0.03, premium, costs, 0.1, scale,
                              ridge.beta * scale, steps=60, validate_every=10, optimizer="lbfgs", smooth_abs=1e-3)
    assert fitted["steps_done"] <= 60 and fitted["status"] in ("budget_exhausted", "lbfgs_terminated", "stationary")
    assert fitted["history"][0]["step"] == 0
    assert fitted["validation_mse"] < fitted["history"][0]["validation_mse"]  # beats the ridge start
    assert fitted["validation_mse"] == min(h["validation_mse"] for h in fitted["history"])
    theta = readout_positions(phi, fitted["beta"], spot)
    exact = cost_metrics(payoff, premium, execution_cashflows(theta, spot, times, 0.03, costs))
    assert exact["net_mse"] == pytest.approx(fitted["validation_mse"])
    with pytest.raises(ValueError):
        fit_cost_readout(phi, spot, payoff, phi, spot, payoff, times, 0.03, premium, costs, 0.1, scale,
                         ridge.beta * scale, optimizer="sgd")


def test_cost_metrics_report_dispersion_and_shortfall_tail():
    payoff = np.array([10.0, 2.0, 4.0, 6.0, 8.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    theta = np.zeros((20, 3))
    spot = np.full((20, 4), 100.0)
    cash = execution_cashflows(theta, spot, np.linspace(0, 1, 4), 0.0, CostSpec(0.01, 0.0))
    metrics = cost_metrics(payoff, 5.0, cash)
    errors = payoff - 5.0
    assert metrics["error_std"] == pytest.approx(errors.std(ddof=1))
    assert metrics["shortfall_cvar95"] == pytest.approx(14.0)  # top 5% of 20 errors = the single largest
    assert metrics["mean_cost"] == 0.0 and metrics["mean_turnover"] == 0.0


def test_cost_transformer_trades_three_assets_for_basket_spot():
    pytest.importorskip("torch")
    from rsmc.cost_transformer import evaluate_cost_transformer, fit_cost_transformer
    from rsmc.transformer_hedging import TransformerHedgeConfig
    rng = np.random.default_rng(306)
    spot = 100.0 * np.exp(np.cumsum(np.concatenate([np.zeros((8, 1, 3)), rng.normal(0, 0.02, (8, 4, 3))], axis=1), axis=1))
    times = np.linspace(0.0, 1.0, 5)
    features = PathFeatures(np.stack([np.broadcast_to(times, (8, 5)), np.log(spot[..., 0] / 100.0)], axis=-1), ("time", "x"))
    payoff = spot[:, -1, :].mean(axis=1) - 95.0
    config = TransformerHedgeConfig(d_model=4, n_heads=2, n_layers=1, window=2, epochs=1, batch_size=4, seed=307)
    fitted = fit_cost_transformer(features, spot, payoff, features, spot, payoff, times, 0.03, 4.0, config,
                                  costs=CostSpec(0.001, 0.0), lam=0.1)
    assert fitted.model.head.out_features == 3
    output = evaluate_cost_transformer(fitted, features, spot, payoff, times, 0.03, batch_size=3)
    assert output["cashflows"]["trades"].shape == (8, 5, 3)
    assert output["errors"].shape == (8,)
    assert np.all(output["cashflows"]["cost"] >= 0)

"""Hand cash flows, gradients and fixed-premium zero-cost consistency."""
import numpy as np
import pytest

from rsmc.cost_hedging import (
    CostSpec, execution_cashflows, execution_cashflows_torch, cost_metrics,
    hedge_readout_features, gain_row_scale, fit_cost_readout,
)
from rsmc.hedging import hedging_features, fit_fixed_premium_hedge, evaluate_hedge


def test_entry_rebalance_liquidation_and_hand_discounted_cashflows():
    # q=(2,-3,1): buy two, sell three, close the remaining short by buying one.
    theta = np.array([[2., -1.]])
    spot = np.array([[100., 110., 120.]])
    original_spot = spot.copy()
    times = np.array([0., 1., 2.])
    rate = np.log(2.)
    cash = execution_cashflows(theta, spot, times, rate, CostSpec(.01, .02, 2.))
    np.testing.assert_array_equal(cash["trades"], [[2., -3., 1.]])
    # Discounted prices are (100,55,30); gains = 2*(-45)-1*(-25)=-65.
    np.testing.assert_allclose(cash["gain"], [-65.])
    np.testing.assert_allclose(cash["proportional_cost"], [.01*(200+165+30)])
    np.testing.assert_allclose(cash["temporary_cost"], [.02/2*(400+495+30)])
    np.testing.assert_allclose(cash["share_turnover"], [6.])
    np.testing.assert_allclose(cash["notional_turnover"], [395.])
    expected = 8.-5.-(-65.)+3.95+9.25
    metrics = cost_metrics(np.array([8., 8.]), 5., {k: np.repeat(v, 2, axis=0) for k,v in cash.items()})
    np.testing.assert_allclose(metrics["errors"], expected)
    assert metrics["net_mse"] == pytest.approx(expected**2)
    np.testing.assert_array_equal(spot, original_spot)


def test_constant_position_costs_only_entry_and_liquidation_and_zero_positions():
    spot = np.array([[100., 102., 105., 108.]])
    times = np.array([0., .25, .5, 1.])
    spec = CostSpec(.001, .002)
    cash = execution_cashflows(np.full((1, 3), 2.), spot, times, .03, spec)
    np.testing.assert_array_equal(cash["trades"], [[2., 0., 0., -2.]])
    assert cash["cost"][0] == pytest.approx((100+108*np.exp(-.03))*(.001*2+.002*4))
    zero = execution_cashflows(np.zeros((1,3)), spot, times, .03, spec)
    for value in zero.values():
        np.testing.assert_array_equal(value, np.zeros_like(value))


def test_torch_matches_numpy_and_autograd_matches_finite_differences():
    torch = pytest.importorskip("torch")
    theta = torch.tensor([[.2, -.1, .4], [.5, .1, -.2]], dtype=torch.float64, requires_grad=True)
    spot = torch.tensor([[100., 101., 99., 104.], [100., 98., 102., 103.]], dtype=torch.float64)
    times = torch.tensor([0., .3, .6, 1.], dtype=torch.float64)
    spec = CostSpec(.001, .002)
    out = execution_cashflows_torch(theta, spot, times, .03, spec)
    reference = execution_cashflows(theta.detach().numpy(), spot.numpy(), times.numpy(), .03, spec)
    for key, value in out.items():
        np.testing.assert_allclose(value.detach().numpy(), reference[key])
    def net_loss(positions):
        cash = execution_cashflows_torch(positions, spot, times, .03, spec)
        error = torch.tensor([3., 4.]) - 2.5 - cash["gain"] + cash["cost"]
        return error.square().mean()
    assert torch.autograd.gradcheck(net_loss, (theta,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_zero_cost_optimizer_stays_at_fixed_premium_ridge_stationary_point():
    rng = np.random.default_rng(81)
    times = np.linspace(0, 1, 5)
    spot = 100*np.exp(np.cumsum(np.column_stack([np.zeros(24), rng.normal(0,.02,(24,4))]),axis=1))
    phi = hedge_readout_features(hedging_features(spot,times), times)
    psi, scale = gain_row_scale(phi, spot, times, .03)
    payoff = rng.normal(5, 2, 24)
    premium = 4.5
    lam = .1
    ridge = fit_fixed_premium_hedge(psi, payoff, premium, lam)
    fitted = fit_cost_readout(phi, spot, payoff, phi, spot, payoff, times, .03,
                             premium, CostSpec(0,0), lam, scale, ridge.beta*scale, steps=10)
    assert fitted["status"] == "stationary"
    assert fitted["steps_done"] == fitted["best_step"] == 0
    np.testing.assert_allclose(fitted["beta"], ridge.beta, rtol=1e-12, atol=1e-12)
    cash = execution_cashflows(phi[:,:-1]@fitted["beta"],spot,times,.03,CostSpec(0,0))
    report = cost_metrics(payoff,premium,cash)
    reference = evaluate_hedge(ridge,psi,payoff)
    np.testing.assert_allclose(report["errors"],reference.errors,atol=1e-12)
    assert fitted["history"][0]["train_objective"] == pytest.approx(
        reference.error_mse + lam/len(payoff)*np.sum((ridge.beta*scale)**2))


def test_positive_cost_readout_selects_unpenalized_validation_checkpoint():
    rng = np.random.default_rng(82)
    times = np.linspace(0,1,4)
    spot = 100*np.exp(np.cumsum(np.column_stack([np.zeros(16),rng.normal(0,.03,(16,3))]),axis=1))
    phi = hedge_readout_features(hedging_features(spot,times),times)
    psi,scale = gain_row_scale(phi,spot,times,.03)
    payoff = np.maximum(spot[:,-1]-100,0)
    premium = payoff.mean()
    ridge=fit_fixed_premium_hedge(psi,payoff,premium,.1)
    fitted=fit_cost_readout(phi,spot,payoff,phi,spot,payoff,times,.03,premium,
                            CostSpec(.001,.001),.1,scale,ridge.beta*scale,steps=3)
    assert fitted["steps_done"] <= 3
    assert fitted["validation_mse"] == min(h["validation_mse"] for h in fitted["history"])
    np.testing.assert_allclose(fitted["beta"]*scale,fitted["gamma"])
    cash=execution_cashflows(phi[:,:-1]@fitted["beta"],spot,times,.03,CostSpec(.001,.001))
    assert cost_metrics(payoff,premium,cash)["net_mse"] == pytest.approx(fitted["validation_mse"])

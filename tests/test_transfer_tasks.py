"""Transfer adapters preserve existing information and multi-asset cash flows."""
import numpy as np
import pytest

from rsmc.cost_hedging import (
    CostSpec, execution_cashflows, execution_cashflows_torch, gain_row_scale,
    hedge_readout_features, fit_cost_readout, readout_positions,
)
from rsmc.hedging import fit_fixed_premium_hedge
from rsmc.transfer_tasks import make_transfer_dataset


def test_multiasset_cashflows_equal_assetwise_accounting_and_gradients():
    torch = pytest.importorskip("torch")
    theta = np.array([[[2., -.5, .2], [-1., .3, -.1]],
                      [[.4, -.2, .3], [-.1, .1, -.2]]])
    spot = np.array([[[100., 80., 120.], [110., 70., 130.], [120., 90., 100.]],
                     [[100., 80., 120.], [105., 90., 115.], [110., 85., 125.]]])
    times = np.array([0., 1., 2.])
    spec = CostSpec(.01, .02, 2.)
    original = spot.copy()
    result = execution_cashflows(theta, spot, times, np.log(2.), spec)
    scalar = [execution_cashflows(theta[:, :, j], spot[:, :, j], times,
                                 np.log(2.), spec) for j in range(3)]
    for key, value in result.items():
        expected = (np.stack([s[key] for s in scalar], axis=-1) if key == "trades"
                    else sum(s[key] for s in scalar))
        np.testing.assert_allclose(value, expected)
    tensor_theta = torch.tensor(theta, requires_grad=True)
    tensor_spot, grid = torch.tensor(spot), torch.tensor(times)
    tensor_result = execution_cashflows_torch(tensor_theta, tensor_spot, grid,
                                              np.log(2.), spec)
    for key, value in result.items():
        np.testing.assert_allclose(tensor_result[key].detach(), value)
    def objective(positions):
        cash = execution_cashflows_torch(positions, tensor_spot, grid, np.log(2.), spec)
        return (3. - cash["gain"] + cash["cost"]).square().mean()
    assert torch.autograd.gradcheck(objective, (tensor_theta,), atol=1e-5, rtol=1e-4)
    zero = execution_cashflows(np.zeros_like(theta), spot, times, .03, spec)
    for value in zero.values():
        np.testing.assert_array_equal(value, np.zeros_like(value))
    np.testing.assert_array_equal(spot, original)


def test_multiasset_fixed_premium_ridge_and_positive_cost_checkpoint():
    data, times, rate, premium, _ = make_transfer_dataset(
        "basket", {"train": 40, "val": 20}, 4, 73)
    train, val = data["train"], data["val"]
    phi = {name: hedge_readout_features(d["features"], times) for name, d in data.items()}
    psi, scale = gain_row_scale(phi["train"], train["spot"], times, rate)
    ridge = fit_fixed_premium_hedge(psi, train["payoff"], premium, .1)
    theta = readout_positions(phi["train"], ridge.beta, train["spot"])
    assert theta.shape == (40, 4, 3)
    cash = execution_cashflows(theta, train["spot"], times, rate, CostSpec(0., 0.))
    np.testing.assert_allclose(cash["gain"], psi @ ridge.beta, atol=1e-12)
    fitted = fit_cost_readout(phi["train"], train["spot"], train["payoff"],
        phi["val"], val["spot"], val["payoff"], times, rate, premium,
        CostSpec(0., 0.), .1, scale, ridge.beta * scale, steps=3)
    assert fitted["status"] == "stationary"
    assert fitted["best_step"] == 0
    np.testing.assert_allclose(fitted["beta"], ridge.beta, atol=1e-12)
    positive = fit_cost_readout(phi["train"], train["spot"], train["payoff"],
        phi["val"], val["spot"], val["payoff"], times, rate, premium,
        CostSpec(.0005, .0005), .1, scale, ridge.beta * scale, steps=3)
    assert positive["validation_mse"] == min(h["validation_mse"] for h in positive["history"])


@pytest.mark.parametrize("task,channels,assets", [
    ("black_scholes", 3, 1), ("heston", 3, 1), ("basket", 5, 3)])
def test_adapter_uses_existing_price_features_traded_prices_and_frozen_premium(task, channels, assets):
    data, times, rate, premium, metadata = make_transfer_dataset(
        task, {"train": 24, "val": 12, "test": 16}, 5, 74)
    assert premium == float(np.float32(data["train"]["payoff"].mean()))
    assert metadata["premium"] == premium
    assert len(set(metadata["dataset_spot_sha256"].values())) == 3
    for split in data.values():
        assert split["features"].values.shape[2] == channels
        assert all("vol" not in name and "signal" not in name
                   for name in split["features"].names)
        phi = hedge_readout_features(split["features"], times)
        beta = np.ones(phi.shape[2] * assets)
        theta = readout_positions(phi, beta, split["spot"])
        actual = execution_cashflows(theta, split["spot"], times, rate, CostSpec(0., 0.))
        psi, _ = gain_row_scale(phi, split["spot"], times, rate)
        np.testing.assert_allclose(actual["gain"], psi @ beta, atol=1e-12)
        monitoring = split["spot"][:, 1:]
        if assets > 1:
            monitoring = monitoring.mean(axis=2)
        np.testing.assert_allclose(split["payoff"], np.exp(-rate*times[-1]) *
                                   np.maximum(monitoring.mean(axis=1)-100., 0.))

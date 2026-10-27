"""Cost-aware use of the unchanged Transformer and shared execution accounting."""
import numpy as np
import pytest

from rsmc.cost_hedging import CostSpec, execution_cashflows_torch
from rsmc.cost_transformer import (
    evaluate_cost_transformer, fit_cost_transformer, transformer_positions,
)
from rsmc.features import PathFeatures
from rsmc.transformer_hedging import (
    TransformerHedgeConfig, _build_model, _make_sliding_causal_mask,
)


def test_cost_positions_match_original_forward_and_are_causal():
    torch = pytest.importorskip("torch")
    torch.manual_seed(71)
    config = TransformerHedgeConfig(d_model=4, n_heads=2, n_layers=2, window=2)
    model = _build_model(torch, torch.nn, 2, 1, config, 3.25)
    model.eval()
    x = torch.randn(3, 5, 2)
    times = torch.linspace(0., 1., 5)
    spot = torch.tensor([[100., 101., 99., 104., 103.]]).repeat(3, 1)
    discounted = spot * torch.exp(-.03 * times)
    increments = torch.diff(discounted, dim=1).unsqueeze(-1)
    mask = _make_sliding_causal_mask(torch, 5, config.window, "cpu")
    with torch.no_grad():
        theta = transformer_positions(model, x, mask)
        cashflows = execution_cashflows_torch(theta, spot, times, .03, CostSpec(0., 0.))
        torch.testing.assert_close(model(x, increments, mask), model.capital + cashflows["gain"])
        assert torch.count_nonzero(cashflows["cost"]) == 0
        changed = x.clone()
        changed[:, 2:] += 100.
        alternative = transformer_positions(model, changed, mask)
        torch.testing.assert_close(theta[:, :2], alternative[:, :2], atol=1e-6, rtol=0)


def test_cost_training_freezes_premium_inputs_and_train_preprocessing():
    pytest.importorskip("torch")
    values = np.arange(48, dtype=float).reshape(6, 4, 2) / 10
    train = PathFeatures(values, ("time", "state"))
    validation = PathFeatures(values + 2, train.names)
    spot = np.full((6, 4), 100.)
    original_spot = spot.copy()
    times = np.linspace(0., 1., 4)
    config = TransformerHedgeConfig(d_model=4, n_heads=2, n_layers=1, window=2,
                                   epochs=1, batch_size=3, seed=72)
    fitted = fit_cost_transformer(
        train, spot, np.full(6, 5.), validation, spot, np.full(6, 5.),
        times, 0., 3., config, costs=CostSpec(.001, .002), lam=.1,
    )
    np.testing.assert_array_equal(spot, original_spot)
    np.testing.assert_allclose(fitted.input_mean, values.astype(np.float32).mean(axis=(0, 1)))
    np.testing.assert_allclose(fitted.input_std, values.astype(np.float32).std(axis=(0, 1)))
    assert float(fitted.model.capital) == fitted.premium == 3.
    assert "capital" not in dict(fitted.model.named_parameters())
    assert fitted.status in ("converged", "budget_exhausted")
    assert len(fitted.history) == 2 and fitted.best_epoch in (0, 1)
    assert np.isfinite(fitted.final_gradient_inf)
    evaluated = evaluate_cost_transformer(fitted, validation, spot, np.full(6, 7.), times, 0., batch_size=2)
    cashflows = evaluated["cashflows"]
    np.testing.assert_allclose(evaluated["errors"], 4. + cashflows["cost"])
    assert evaluated["metrics"]["mean_error"] >= 4.
    assert evaluated["metrics"]["net_mse"] == pytest.approx(np.mean(evaluated["errors"]**2))
    assert all(value >= 0 for value in fitted.timings.values())

"""Fixed-premium and timing checks for the existing Transformer hedge."""

import numpy as np
import pytest

from rsmc.features import PathFeatures
from rsmc.transformer_hedging import (
    TransformerHedgeConfig,
    _build_model,
    _make_sliding_causal_mask,
    evaluate_transformer_hedge,
    fit_transformer_hedge,
)


def test_explicit_premium_and_training_only_preprocessing():
    torch = pytest.importorskip("torch")
    values = np.arange(48, dtype=float).reshape(6, 4, 2) / 10
    train = PathFeatures(values, ("time", "state"))
    validation = PathFeatures(values + 100, train.names)
    increments = np.zeros((6, 3))
    payoff = np.full(6, 5.0)
    config = TransformerHedgeConfig(
        d_model=4, n_heads=2, n_layers=1, window=2,
        epochs=1, batch_size=3, seed=19,
    )
    fitted = fit_transformer_hedge(
        train, increments, payoff, validation, increments, payoff, config,
        initial_capital=3.0,
    )
    np.testing.assert_allclose(fitted.input_mean, values.astype(np.float32).mean(axis=(0, 1)))
    np.testing.assert_allclose(fitted.input_std, values.astype(np.float32).std(axis=(0, 1)))
    assert "capital" not in dict(fitted.model.named_parameters())
    assert not fitted.model.capital.requires_grad
    assert fitted.validation_mse == 4.0
    report = evaluate_transformer_hedge(fitted, validation, increments, np.full(6, 7.0))
    assert report.capital == 3.0
    np.testing.assert_array_equal(report.errors, np.full(6, 4.0))
    assert report.mean_error == 4.0
    assert report.error_mse == 16.0


def test_transformer_uses_only_pre_gain_information():
    torch = pytest.importorskip("torch")
    torch.manual_seed(21)
    config = TransformerHedgeConfig(d_model=4, n_heads=2, n_layers=2, window=2)
    model = _build_model(torch, torch.nn, 2, 1, config, initial_capital=3.0)
    model.eval()
    values = torch.randn(3, 5, 2)
    changed = values.clone()
    # Only the gain over (t_1,t_2] is nonzero; information from t_2 onwards
    # must have no effect on its position.
    changed[:, 2:, :] += 100
    increments = torch.zeros(3, 4, 1)
    increments[:, 1, 0] = 1
    mask = _make_sliding_causal_mask(torch, 5, config.window, "cpu")
    with torch.no_grad():
        original = model(values, increments, mask)
        perturbed = model(changed, increments, mask)
    torch.testing.assert_close(original, perturbed, rtol=0, atol=1e-6)

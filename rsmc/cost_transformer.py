"""Cost-aware training of the existing causal Transformer, without new inputs."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import numpy as np

from .cost_hedging import CostSpec, cost_metrics, execution_cashflows, execution_cashflows_torch
from .features import PathFeatures
from .transformer_hedging import (
    TransformerHedgeConfig, _build_model, _make_sliding_causal_mask,
    _standardise, _torch_modules,
)


def transformer_positions(model, x, mask):
    """The existing model's positions, using only features through each trade time.

    Returns shape (n, N) for one traded asset and (n, N, n_assets) otherwise.
    """
    import torch
    encoded = model.encoder(model.input_projection(x), mask=mask)
    state = torch.cat([encoded[:, :-1, :], x[:, :-1, :]], dim=-1)
    return model.head(state).squeeze(-1)


@dataclass
class CostTransformerFit:
    model: object
    input_mean: np.ndarray
    input_std: np.ndarray
    config: TransformerHedgeConfig
    premium: float
    costs: CostSpec
    validation_mse: float
    best_epoch: int
    history: list[dict]
    status: str
    final_gradient_inf: float
    timings: dict[str, float]
    n_parameters: int


def fit_cost_transformer(
    features_train: PathFeatures, spot_train: np.ndarray, payoff_train: np.ndarray,
    features_val: PathFeatures, spot_val: np.ndarray, payoff_val: np.ndarray,
    times: np.ndarray, rate: float, premium: float, config: TransformerHedgeConfig,
    *, costs: CostSpec, lam: float = .1,
) -> CostTransformerFit:
    """Fixed-epoch AdamW with explicit head penalty, validation-only selection.

    Loss is mean squared net error plus ``lam/n_train * ||head||^2``. The
    penalty includes head weights and bias, but neither the fixed premium nor
    an additional weight-decay penalty. Epoch zero is an eligible checkpoint.
    The reported convergence diagnostic is the full training gradient at the
    final iterate, before loading the validation-selected checkpoint.
    """
    overall_start = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    if config.epochs < 1 or config.batch_size < 1 or config.window < 1:
        raise ValueError("Epochs, batch size and causal window must be positive.")
    if config.d_model % config.n_heads:
        raise ValueError("d_model must be divisible by n_heads.")
    if not np.isfinite(premium) or not np.isfinite(lam) or lam < 0:
        raise ValueError("Premium and nonnegative penalty must be finite.")
    torch.manual_seed(config.seed)
    train_values = features_train.values.astype(np.float32)
    mean = train_values.reshape(-1, train_values.shape[-1]).mean(axis=0)
    std = train_values.reshape(-1, train_values.shape[-1]).std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    x_train = torch.as_tensor(_standardise(train_values, mean, std))
    x_val = torch.as_tensor(_standardise(features_val.values, mean, std))
    s_train = torch.as_tensor(spot_train, dtype=torch.float32)
    s_val = torch.as_tensor(spot_val, dtype=torch.float32)
    y_train = torch.as_tensor(payoff_train, dtype=torch.float32)
    y_val = torch.as_tensor(payoff_val, dtype=torch.float32)
    time_grid = torch.as_tensor(times, dtype=torch.float32, device=config.device)
    n_train = len(y_train)
    n_assets = 1 if s_train.ndim == 2 else int(s_train.shape[2])
    model = _build_model(torch, nn, x_train.shape[-1], n_assets, config, premium).to(config.device)
    mask = _make_sliding_causal_mask(torch, x_train.shape[1], config.window, config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    loader = DataLoader(TensorDataset(x_train, s_train, y_train), batch_size=config.batch_size, shuffle=True)
    timings = {"setup_s": time.perf_counter() - overall_start, "training_s": 0.0,
               "validation_s": 0.0, "convergence_check_s": 0.0}

    def head_penalty():
        return (lam / n_train) * sum(p.square().sum() for p in model.head.parameters())

    def net_error(x, spot, payoff):
        theta = transformer_positions(model, x.to(config.device), mask)
        cf = execution_cashflows_torch(theta, spot.to(config.device), time_grid, rate, costs)
        return payoff.to(config.device) - model.capital - cf["gain"] + cf["cost"]

    def validate():
        started = time.perf_counter()
        model.eval()
        sum_squares = 0.0
        with torch.no_grad():
            for start in range(0, len(y_val), 1000):
                stop = start + 1000
                errors = net_error(x_val[start:stop], s_val[start:stop], y_val[start:stop])
                sum_squares += float(errors.square().sum().cpu())
        mse = sum_squares / len(y_val)
        if not np.isfinite(mse):
            raise FloatingPointError("Nonfinite Transformer validation net MSE.")
        timings["validation_s"] += time.perf_counter() - started
        return mse

    best_val = validate()
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "validation_mse": best_val}]
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        accumulated_objective = 0.0
        for batch_x, batch_spot, batch_payoff in loader:
            optimizer.zero_grad()
            errors = net_error(batch_x, batch_spot, batch_payoff)
            loss = errors.square().mean() + head_penalty()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite cost-aware Transformer training objective.")
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite cost-aware Transformer gradient.")
            optimizer.step()
            accumulated_objective += float(loss.detach().cpu()) * len(batch_payoff)
        timings["training_s"] += time.perf_counter() - started
        val_mse = validate()
        history.append({"epoch": epoch, "mean_batch_objective": accumulated_objective / n_train,
                        "validation_mse": val_mse})
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            best_state = copy.deepcopy(model.state_dict())

    # Accumulated full-data gradient, not a minibatch norm, for honest status.
    started = time.perf_counter()
    model.eval()
    model.zero_grad()
    for start in range(0, n_train, config.batch_size):
        stop = start + config.batch_size
        errors = net_error(x_train[start:stop], s_train[start:stop], y_train[start:stop])
        (errors.square().sum() / n_train).backward()
    head_penalty().backward()
    gradient_inf = max(float(p.grad.detach().abs().max().cpu()) for p in model.parameters() if p.grad is not None)
    if not np.isfinite(gradient_inf):
        raise FloatingPointError("Nonfinite final Transformer convergence diagnostic.")
    model.zero_grad()
    timings["convergence_check_s"] = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    timings["total_s"] = time.perf_counter() - overall_start
    return CostTransformerFit(
        model, mean, std, config, float(premium), costs, best_val, best_epoch, history,
        "converged" if gradient_inf <= 1e-5 else "budget_exhausted", gradient_inf,
        timings, sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def evaluate_cost_transformer(
    fitted: CostTransformerFit, features: PathFeatures, spot: np.ndarray,
    payoff: np.ndarray, times: np.ndarray, rate: float, *, batch_size: int = 1000,
) -> dict:
    """Apply saved training normalization and return unrecentered net errors."""
    torch, _, _, _ = _torch_modules()
    batch_size = min(batch_size, 1000)
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    model = fitted.model
    model.eval()
    mask = _make_sliding_causal_mask(torch, features.values.shape[1], fitted.config.window, fitted.config.device)
    positions = []
    with torch.no_grad():
        for start in range(0, len(payoff), batch_size):
            values = _standardise(features.values[start:start + batch_size], fitted.input_mean, fitted.input_std)
            x = torch.as_tensor(values, device=fitted.config.device)
            positions.append(transformer_positions(model, x, mask).cpu().numpy())
    theta = np.concatenate(positions)
    cashflows = execution_cashflows(theta, spot, times, rate, fitted.costs)
    metrics = cost_metrics(payoff, fitted.premium, cashflows)
    errors = metrics.pop("errors")
    return {"metrics": metrics, "cashflows": cashflows, "errors": errors}

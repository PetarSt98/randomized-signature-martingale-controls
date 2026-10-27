"""Small causal Transformer benchmark for observed-filtration hedging.

This module is intentionally optional: it imports PyTorch only inside the fitting
path, so the rest of the project remains usable when ``torch`` is not installed.
The model is a compact encoder-only Transformer with a sliding causal attention
mask. It is trained directly on the quadratic hedging objective

    mean((H - c - sum_k theta_k dS~_{k+1})**2).

No pretrained models, no decoder, and no language-model machinery are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np

from .features import PathFeatures
from .hedging import HedgeReport


def transformer_available() -> bool:
    """Whether PyTorch can be imported in the current environment."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class TransformerHedgeConfig:
    """Training and architecture knobs for the small causal Transformer hedge."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    window: int = 16
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 256
    seed: int = 0
    device: str = "cpu"


@dataclass
class FittedTransformerHedge:
    """A fitted Transformer hedge plus input standardisation metadata."""

    model: object
    input_mean: np.ndarray
    input_std: np.ndarray
    config: TransformerHedgeConfig
    validation_mse: float
    fit_s: float
    n_parameters: int


def _torch_modules():
    """Import PyTorch lazily with a clear dependency error."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(
            "Transformer hedge benchmark requires a working PyTorch install. "
            "Install or repair torch to run the transformer rows."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _standardise(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _as_vector_increments(increments: np.ndarray) -> np.ndarray:
    if increments.ndim == 2:
        return increments[:, :, None]
    if increments.ndim == 3:
        return increments
    raise ValueError(f"increments must have shape (n, N) or (n, N, r), got {increments.shape}.")


def _make_sliding_causal_mask(torch, seq_len: int, window: int, device: str):
    """Float attention mask: each time sees only itself and the previous window."""
    positions = torch.arange(seq_len, device=device)
    query = positions[:, None]
    key = positions[None, :]
    blocked = (key > query) | ((query - key) > window)
    mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    mask = mask.masked_fill(blocked, float("-inf"))
    return mask


def _build_model(
    torch,
    nn,
    input_dim: int,
    output_dim: int,
    config: TransformerHedgeConfig,
    initial_capital: float,
):
    class CausalTransformerHedge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=4 * config.d_model,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
            self.head = nn.Linear(config.d_model + input_dim, output_dim)
            self.register_buffer(
                "capital",
                torch.tensor(float(initial_capital), dtype=torch.float32),
            )

        def forward(self, x, increments, mask):
            encoded = self.encoder(self.input_projection(x), mask=mask)
            state = torch.cat([encoded[:, :-1, :], x[:, :-1, :]], dim=-1)
            theta = self.head(state)
            gain = (theta * increments).sum(dim=(1, 2))
            return self.capital + gain

    return CausalTransformerHedge()


def fit_transformer_hedge(
    features_train: PathFeatures,
    increments_train: np.ndarray,
    payoff_train: np.ndarray,
    features_val: PathFeatures,
    increments_val: np.ndarray,
    payoff_val: np.ndarray,
    config: TransformerHedgeConfig,
    *,
    initial_capital: float | None = None,
) -> FittedTransformerHedge:
    """Train the causal hedge and select by unpenalised validation MSE.

    ``initial_capital`` is fixed throughout training and evaluation. If omitted,
    retain the historical convention of using the training payoff mean. Supply
    the same premium to every method for a fixed-premium comparison.
    """
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    torch.manual_seed(config.seed)
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if config.window <= 0:
        raise ValueError("window must be positive.")

    train_values = features_train.values.astype(np.float32)
    mean = train_values.reshape(-1, train_values.shape[-1]).mean(axis=0)
    std = train_values.reshape(-1, train_values.shape[-1]).std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    x_train = torch.as_tensor(_standardise(features_train.values, mean, std), dtype=torch.float32)
    inc_train = torch.as_tensor(_as_vector_increments(increments_train), dtype=torch.float32)
    y_train = torch.as_tensor(payoff_train.astype(np.float32), dtype=torch.float32)
    x_val = torch.as_tensor(_standardise(features_val.values, mean, std), dtype=torch.float32, device=config.device)
    inc_val = torch.as_tensor(_as_vector_increments(increments_val), dtype=torch.float32, device=config.device)
    y_val = torch.as_tensor(payoff_val.astype(np.float32), dtype=torch.float32, device=config.device)

    output_dim = inc_train.shape[2]
    initial_capital = float(np.mean(payoff_train)) if initial_capital is None else float(initial_capital)
    if not np.isfinite(initial_capital):
        raise ValueError("initial_capital must be finite.")
    model = _build_model(
        torch,
        nn,
        x_train.shape[-1],
        output_dim,
        config,
        initial_capital,
    ).to(config.device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(x_train, inc_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    start = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        for batch_x, batch_inc, batch_y in loader:
            batch_x = batch_x.to(config.device)
            batch_inc = batch_inc.to(config.device)
            batch_y = batch_y.to(config.device)
            mask = _make_sliding_causal_mask(torch, batch_x.shape[1], config.window, config.device)
            prediction = model(batch_x, batch_inc, mask)
            loss = torch.mean((batch_y - prediction) ** 2)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_mask = _make_sliding_causal_mask(torch, x_val.shape[1], config.window, config.device)
            val_prediction = model(x_val, inc_val, val_mask)
            val_mse = float(torch.mean((y_val - val_prediction) ** 2).cpu())
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())

    fit_s = time.perf_counter() - start
    model.load_state_dict(best_state)
    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return FittedTransformerHedge(
        model=model,
        input_mean=mean,
        input_std=std,
        config=config,
        validation_mse=best_val,
        fit_s=fit_s,
        n_parameters=n_parameters,
    )


def evaluate_transformer_hedge(
    fitted: FittedTransformerHedge,
    features: PathFeatures,
    increments: np.ndarray,
    payoff: np.ndarray,
) -> HedgeReport:
    """Evaluate a fitted Transformer hedge on independent paths."""
    torch, _, _, _ = _torch_modules()
    model = fitted.model
    device = fitted.config.device
    x = torch.as_tensor(
        _standardise(features.values, fitted.input_mean, fitted.input_std),
        dtype=torch.float32,
        device=device,
    )
    inc = torch.as_tensor(_as_vector_increments(increments), dtype=torch.float32, device=device)
    mask = _make_sliding_causal_mask(torch, x.shape[1], fitted.config.window, device)

    model.eval()
    with torch.no_grad():
        wealth = model(x, inc, mask).cpu().numpy()

    error = payoff - wealth
    error_var = float(error.var(ddof=1))
    payoff_var = float(payoff.var(ddof=1))
    capital = float(model.capital.detach().cpu())
    return HedgeReport(
        n_eval=int(payoff.size),
        price=float(payoff.mean()),
        capital=capital,
        payoff_std=float(np.sqrt(payoff_var)),
        error_std=float(np.sqrt(error_var)),
        rmse=float(np.sqrt(np.mean(error**2))),
        error_mse=float(np.mean(error**2)),
        error_var=error_var,
        variance_reduction=payoff_var / error_var if error_var > 0 else np.inf,
        mean_error=float(error.mean()),
        mean_error_se=float(error.std(ddof=1) / np.sqrt(payoff.size)),
        errors=error,
    )

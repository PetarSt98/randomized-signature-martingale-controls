"""Execution cash flows and bounded fixed-feature readout optimization.

Positions are shares held over each next interval. Costs are cash payments at
the observed execution price; they never change subsequent simulated prices.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .features import PathFeatures
from .control import control_features
from .hedging import discounted_traded_increments
from .reservoir import readout_features


@dataclass(frozen=True)
class CostSpec:
    """Dimensionless per-trade rates, with q_ref measured in shares.

    At price S an order q costs S*(kappa*abs(q) + eta*q*q/q_ref).
    eta is the temporary-cost rate at reference order q_ref. This is a
    discrete per-execution model: no division by a trading time interval.
    """

    kappa: float
    eta: float
    q_ref: float = 1.0

    def __post_init__(self):
        if not np.isfinite([self.kappa, self.eta, self.q_ref]).all():
            raise ValueError("Cost coefficients and reference quantity must be finite.")
        if self.kappa < 0 or self.eta < 0 or self.q_ref <= 0:
            raise ValueError("Cost coefficients must be nonnegative and q_ref positive.")


def execution_cashflows(theta, spot, times, rate, costs: CostSpec):
    """Numpy pathwise accounting, including entry and terminal liquidation."""
    theta = np.asarray(theta, dtype=float)
    spot = np.asarray(spot, dtype=float)
    times = np.asarray(times, dtype=float)
    if theta.ndim not in (2, 3) or spot.shape != (theta.shape[0], theta.shape[1] + 1, *theta.shape[2:]):
        raise ValueError("Expected matching theta (paths,N[,assets]) and spot (paths,N+1[,assets]).")
    if times.shape != (spot.shape[1],) or theta.shape[1] == 0:
        raise ValueError("Time grid must cover every execution, including liquidation.")
    traded = np.concatenate([theta[:, :1], np.diff(theta, axis=1), -theta[:, -1:]], axis=1)
    discount = np.exp(-rate * times)[None, :]
    discounted_spot = spot * (discount if spot.ndim == 2 else discount[:, :, None])
    axes = 1 if theta.ndim == 2 else (1, 2)
    proportional = costs.kappa * (discounted_spot * np.abs(traded)).sum(axis=axes)
    temporary = costs.eta / costs.q_ref * (discounted_spot * traded**2).sum(axis=axes)
    return {
        "gain": (theta * np.diff(discounted_spot, axis=1)).sum(axis=axes),
        "proportional_cost": proportional,
        "temporary_cost": temporary,
        "cost": proportional + temporary,
        "share_turnover": np.abs(traded).sum(axis=axes),
        "notional_turnover": (discounted_spot * np.abs(traded)).sum(axis=axes),
        "trades": traded,
    }


def execution_cashflows_torch(theta, spot, times, rate, costs: CostSpec):
    """Differentiable accounting with the same units and signs as the numpy path."""
    import torch

    traded = torch.cat([theta[:, :1], torch.diff(theta, dim=1), -theta[:, -1:]], dim=1)
    discount = torch.exp(-rate * times)[None, :]
    discounted_spot = spot * (discount if spot.ndim == 2 else discount[:, :, None])
    axes = 1 if theta.ndim == 2 else (1, 2)
    proportional = costs.kappa * (discounted_spot * traded.abs()).sum(dim=axes)
    temporary = costs.eta / costs.q_ref * (discounted_spot * traded.square()).sum(dim=axes)
    return {
        "gain": (theta * torch.diff(discounted_spot, dim=1)).sum(dim=axes),
        "proportional_cost": proportional,
        "temporary_cost": temporary,
        "cost": proportional + temporary,
        "share_turnover": traded.abs().sum(dim=axes),
        "notional_turnover": (discounted_spot * traded.abs()).sum(dim=axes),
        "trades": traded,
    }


def cost_metrics(payoff, premium, cashflows):
    """Unpenalized, unrecentered discounted net-error and turnover diagnostics."""
    error = np.asarray(payoff) - premium - cashflows["gain"] + cashflows["cost"]
    # Positive error is a shortfall: payoff plus costs exceeded premium plus gains.
    tail = np.sort(error)[-max(1, int(np.ceil(0.05 * error.size))):]
    metrics = {
        "net_mse": float(np.mean(error**2)),
        "net_rmse": float(np.sqrt(np.mean(error**2))),
        "mean_error": float(np.mean(error)),
        "mean_error_se": float(np.std(error, ddof=1) / np.sqrt(error.size)),
        "error_std": float(np.std(error, ddof=1)) if error.size > 1 else 0.0,
        "shortfall_cvar95": float(np.mean(tail)),
        "mean_gain": float(np.mean(cashflows["gain"])),
        "mean_turnover": float(np.mean(cashflows["share_turnover"])),
        "mean_notional_turnover": float(np.mean(cashflows["notional_turnover"])),
        "mean_proportional_cost": float(np.mean(cashflows["proportional_cost"])),
        "mean_temporary_cost": float(np.mean(cashflows["temporary_cost"])),
        "mean_cost": float(np.mean(cashflows["cost"])),
        "premium": float(premium),
        "n_eval": int(error.size),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite evaluation metrics.")
    return {**metrics, "errors": error}


def hedge_readout_features(features: PathFeatures, times, reservoir=None, *, nonlinear_features=None):
    """Causal [1,t,current non-time state,h], with optional separate nonlinear inputs."""
    current = features.values[:, :, 1:] if features.names[0] == "time" else features.values
    if nonlinear_features is not None:
        if reservoir is None or nonlinear_features.values.shape[:2] != features.values.shape[:2]:
            raise ValueError("Nonlinear inputs need a feature map and matching paths/dates.")
    inputs = features if nonlinear_features is None else nonlinear_features
    state = current if reservoir is None else np.concatenate([current, reservoir.run(inputs)], axis=-1)
    return readout_features(state, times)


def gain_row_scale(phi, spot, times, rate):
    """Training gain rows and RMS scaling; never call on validation to fit scales."""
    increments = discounted_traded_increments(spot, times, rate)
    psi = control_features(phi, increments)
    scale = np.sqrt(np.mean(psi**2, axis=0))
    return psi, np.where(scale < 1e-12, 1.0, scale)


def readout_positions(phi, beta, spot):
    """Predictable positions using the existing asset-major readout blocks."""
    beta = np.asarray(beta)
    if spot.ndim == 2:
        return phi[:, :-1] @ beta
    return phi[:, :-1] @ beta.reshape(spot.shape[2], phi.shape[2]).T


def fit_cost_readout(
    phi_train, spot_train, payoff_train, phi_val, spot_val, payoff_val,
    times, rate, premium, costs: CostSpec, lam, scale, init_gamma,
    *, steps=120, lr=.01, grad_tol=1e-5, validate_every=1, dtype="float64",
    optimizer="adamw", smooth_abs=0.0,
):
    """Bounded full-batch optimization of net MSE + (lam/n_train)*||gamma||².

    ``optimizer="adamw"`` takes ``steps`` AdamW steps at learning rate ``lr``.
    ``optimizer="lbfgs"`` runs torch L-BFGS with strong-Wolfe line search for at
    most ``steps`` iterations, in chunks of ``validate_every`` iterations between
    validations. ``smooth_abs`` > 0 replaces |q| in the *training* proportional
    cost by sqrt(q² + eps²) - eps (shares), a convex smooth lower bound, so
    quasi-Newton steps are well defined; validation always uses exact costs.

    No ridge fit is hidden here: the caller supplies and counts initialization.
    beta=gamma/scale, with training-only gain-row RMS scales held fixed. Every
    validated checkpoint (including initialization) is selected by unpenalized
    validation net MSE; ``validate_every`` > 1 validates only every that many
    steps plus the final iterate. Gradient stationarity concerns the final
    optimization iterate, not a global optimum, and need not characterize the
    validation-selected iterate.
    """
    import torch

    started = time.perf_counter()
    if steps < 0 or lam < 0 or lr <= 0 or grad_tol < 0 or validate_every < 1 or smooth_abs < 0:
        raise ValueError("Invalid optimizer limits or regularization.")
    if optimizer not in ("adamw", "lbfgs"):
        raise ValueError("optimizer must be 'adamw' or 'lbfgs'.")
    scale = np.asarray(scale, dtype=float)
    init_gamma = np.asarray(init_gamma, dtype=float)
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isfinite(init_gamma).all():
        raise ValueError("Scale and initializer must be finite; scales must be positive.")
    dtype = getattr(torch, dtype)
    convert = lambda value: torch.as_tensor(np.ascontiguousarray(value), dtype=dtype)
    n_assets = 1 if np.ndim(spot_train) == 2 else np.shape(spot_train)[2]
    scalar = np.ndim(spot_train) == 2
    scale_tensor = convert(scale)
    # Predictable prefix only; scalar readouts fold the gain-row scale into phi once.
    train_phi = convert(phi_train[:, :-1])
    val_phi = convert(phi_val[:, :-1])
    if scalar:
        train_phi, val_phi = train_phi / scale_tensor, val_phi / scale_tensor
    train_spot, val_spot = convert(spot_train), convert(spot_val)
    train_payoff, val_payoff, grid = convert(payoff_train), convert(payoff_val), convert(times)
    gamma = torch.nn.Parameter(convert(init_gamma).clone())
    def positions(phi):
        return phi @ gamma if scalar else phi @ (gamma / scale_tensor).reshape(n_assets, -1).T
    train_discounted = train_spot * (torch.exp(-rate * grid)[None, :] if scalar else torch.exp(-rate * grid)[None, :, None])
    axes = 1 if scalar else (1, 2)
    def training_objective():
        theta = positions(train_phi)
        traded = torch.cat([theta[:, :1], torch.diff(theta, dim=1), -theta[:, -1:]], dim=1)
        magnitude = traded.abs() if smooth_abs == 0 else torch.sqrt(traded.square() + smooth_abs**2) - smooth_abs
        cost = (costs.kappa * (train_discounted * magnitude).sum(dim=axes)
                + costs.eta / costs.q_ref * (train_discounted * traded.square()).sum(dim=axes))
        gain = (theta * torch.diff(train_discounted, dim=1)).sum(dim=axes)
        error = train_payoff - premium - gain + cost
        train_mse = error.square().mean()
        return train_mse + (lam / len(payoff_train)) * gamma.square().sum(), train_mse
    use_lbfgs = optimizer == "lbfgs"
    if use_lbfgs:
        optimizer = torch.optim.LBFGS([gamma], lr=1.0, max_iter=validate_every, history_size=20,
                                      line_search_fn="strong_wolfe", tolerance_grad=grad_tol, tolerance_change=1e-14)
        def closure():
            optimizer.zero_grad(set_to_none=True)
            objective, _ = training_objective()
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training objective inside L-BFGS line search.")
            objective.backward()
            return objective
    else:
        optimizer = torch.optim.AdamW([gamma], lr=lr, weight_decay=0.)
    setup_s = time.perf_counter() - started
    validation_s = 0.
    optimization_s = 0.
    history = []
    best_score, best_gamma, best_step = np.inf, None, None
    status = "budget_exhausted"
    step = 0
    terminated = False  # L-BFGS stopped inside a chunk (tolerance or failed line search)
    while step <= steps:
        stage = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        objective, train_mse = training_objective()
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite training objective at step {step}.")
        objective.backward()
        gradient_inf = float(gamma.grad.abs().max())
        if not np.isfinite(gradient_inf):
            raise FloatingPointError(f"Nonfinite gradient at step {step}.")
        optimization_s += time.perf_counter() - stage
        stationary = gradient_inf <= grad_tol
        if use_lbfgs or step % validate_every == 0 or step == steps or stationary:
            stage = time.perf_counter()
            with torch.no_grad():
                val_cash = execution_cashflows_torch(positions(val_phi), val_spot, grid, rate, costs)
                val_error = val_payoff - premium - val_cash["gain"] + val_cash["cost"]
                validation_mse = float(val_error.square().mean())
            validation_s += time.perf_counter() - stage
            if not np.isfinite(validation_mse):
                raise FloatingPointError(f"Nonfinite validation MSE at step {step}.")
            history.append(dict(step=step, train_mse=float(train_mse.detach()),
                                train_objective=float(objective.detach()),
                                validation_mse=validation_mse, gradient_inf=gradient_inf))
            if validation_mse < best_score:
                best_score, best_gamma, best_step = validation_mse, gamma.detach().numpy().copy(), step
        if stationary:
            status = "stationary"
            break
        if terminated:
            status = "lbfgs_terminated"
            break
        if step >= steps:
            break
        stage = time.perf_counter()
        if use_lbfgs:
            before = optimizer.state[gamma].get("n_iter", 0) if optimizer.state else 0
            optimizer.step(closure)
            done = int(optimizer.state[gamma]["n_iter"] - before)
            terminated = done < validate_every
            step = min(step + max(done, 1), steps)
        else:
            optimizer.step()
            step += 1
        optimization_s += time.perf_counter() - stage
    return {
        "beta": best_gamma / scale, "gamma": best_gamma, "best_step": best_step,
        "validation_mse": best_score, "status": status, "gradient_inf": gradient_inf,
        "steps_done": step, "history": history, "fit_s": time.perf_counter() - started,
        "setup_s": setup_s, "optimization_s": optimization_s, "validation_s": validation_s,
    }

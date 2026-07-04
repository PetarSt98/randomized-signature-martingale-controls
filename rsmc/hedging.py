"""Partial-information quadratic hedging with reservoir / signature readouts.

Phase II of the project. The trader must hedge a discounted payoff ``H`` by
trading only the discounted stock ``S~_k = e^{-r t_k} S_k``, using a strategy that
is *predictable with respect to the observed filtration*

    G_{t_k} = sigma( (S_i, Y_i) : i <= k ),     Y_k = V_k + eta * eps_k,

where the true variance ``V`` is hidden and only the noisy signal ``Y`` is seen.
The hedging wealth is

    c + sum_{k=0}^{N-1} theta_k * dS~_{k+1},     theta_k = A_beta(h_k, t_k),

with ``h_k`` a *causal* state (explicit observed features, a frozen reservoir, or a
truncated signature) and ``A_beta`` a trained linear readout.

Why this reuses the pricing code
--------------------------------
``S~`` is a Q-martingale in the full filtration F (log-Euler keeps
``E[S~_{k+1} | F_k] = S~_k`` exactly). Since ``G_k subset F_k``, the tower property
gives ``E[dS~_{k+1} | G_k] = 0`` as well, so ``dS~`` is a G-martingale increment.
Collapsing each path into one regression row,

    Psi_i = sum_{k=0}^{N-1} phi_{i,k} * dS~_{i,k+1},

the best linear hedge is exactly the ridge problem

    min_{c, beta}  sum_i ( H_i - c - beta . Psi_i )^2 + lambda ||beta||^2,

i.e. the pricing control-variate regression with the simulator increment ``dM``
replaced by the traded increment ``dS~`` and the intercept *kept* as the initial
capital ``c``. For reservoir and signature hedges the readout is augmented with
the current observed state,

    phi_k = [1, t_k, x_k, h_k],

so it has the same direct current-state information as the Transformer hedge
head. Only the increment, target and reporting differ from the pricing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import RidgeMartingaleControl, control_features
from .features import PathFeatures
from .reservoir import RandomizedSignatureReservoir, readout_features
from .signature import _iterate_running_signature, signature_dim


def observe_variance(variance: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    """Noisy direct observation ``Y_k = V_k + eta * eps_k`` of the hidden variance.

    Parameters
    ----------
    variance:
        Hidden variance paths ``V``, shape ``(n_paths, N + 1)`` for one factor
        or ``(n_paths, N + 1, n_assets)`` for multiple variance factors.
    eta:
        Observation-noise standard deviation (``eta = 0`` returns ``V`` exactly).
    rng:
        Generator for the observation noise (kept independent of the path noise so
        the signal is reproducible and not correlated with the hidden shocks).
    """
    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}.")
    signal = variance + eta * rng.standard_normal(variance.shape)
    signal[:, 0] = variance[:, 0]                    # V_0 is known at inception
    return signal


def discounted_traded_increments(spot: np.ndarray, times: np.ndarray, rate: float) -> np.ndarray:
    """Discounted traded-price increments ``dS~``.

    Parameters
    ----------
    spot:
        Asset prices ``S``, shape ``(n_paths, N + 1)`` for one stock or
        ``(n_paths, N + 1, n_assets)`` for multiple traded stocks.
    times:
        Time grid, shape ``(N + 1,)``.
    rate:
        Risk-free rate used to discount the traded price.

    Returns
    -------
    np.ndarray
        Discounted price increments, shape ``(n_paths, N)`` or
        ``(n_paths, N, n_assets)``. These are the G-martingale increments the
        hedge trades against.
    """
    spot = np.asarray(spot, dtype=float)
    if spot.ndim == 2:
        discount = np.exp(-rate * times)[None, :]
    elif spot.ndim == 3:
        discount = np.exp(-rate * times)[None, :, None]
    else:
        raise ValueError(f"spot must have shape (n, N + 1) or (n, N + 1, r), got {spot.shape}.")
    discounted = spot * discount
    return np.diff(discounted, axis=1)


def hedging_features(
    spot: np.ndarray,
    times: np.ndarray,
    vol: np.ndarray | None = None,
    vol_name: str = "vol",
) -> PathFeatures:
    """Causal, observable hedging features, each starting at (approximately) zero.

    The base channels are observable from the price history alone:

        1. time             tau_k = t_k / T
        2. log-price        log(S_k / S_0)
        3. running average  log(A_k / S_0),  A_k = mean(S_0, ..., S_k)

    An optional fourth channel ``vol`` carries the volatility information used by
    the hedge: the true variance for the full-information oracle, or the noisy
    observed signal for the admissible observed-filtration hedge. ``vol`` must be
    causal; pass it pre-centred (e.g. ``V - V_0`` or ``Y - V_0``) so it also starts
    near zero. The reservoir and signature consume the increments of these
    channels, so a constant per-channel shift does not affect them.
    """
    spot = np.asarray(spot, dtype=float)
    if spot.ndim != 2:
        raise ValueError(f"spot must have shape (n_paths, N + 1), got {spot.shape}.")

    n_paths, n_points = spot.shape
    s0 = spot[:, 0:1]
    log_price = np.log(spot / s0)
    counts = np.arange(1, n_points + 1)
    running_average = np.log(np.cumsum(spot, axis=1) / counts / s0)
    tau = np.broadcast_to(times / times[-1], (n_paths, n_points))

    channels = [tau, log_price, running_average]
    names = ["time", "log_price", "running_average"]
    if vol is not None:
        vol = np.asarray(vol, dtype=float)
        if vol.shape != spot.shape:
            raise ValueError(f"vol must match spot shape {spot.shape}, got {vol.shape}.")
        channels.append(vol)
        names.append(vol_name)

    return PathFeatures(values=np.stack(channels, axis=-1), names=tuple(names))


def multi_asset_hedging_features(
    spot: np.ndarray,
    times: np.ndarray,
    vol: np.ndarray | None = None,
    vol_name: str = "signal",
    weights: np.ndarray | tuple[float, ...] | None = None,
) -> PathFeatures:
    """Causal three-asset basket hedging features, exactly ``d = 8`` with ``vol``.

    Channels are time, three log-stock ratios, three volatility/signal channels,
    and the running basket-average log ratio. With ``vol=None`` this returns the
    price-only baseline channels ``d = 5``.
    """
    spot = np.asarray(spot, dtype=float)
    if spot.ndim != 3 or spot.shape[2] != 3:
        raise ValueError(f"spot must have shape (n_paths, N + 1, 3), got {spot.shape}.")

    n_paths, n_points, n_assets = spot.shape
    if weights is None:
        weight_array = np.full(n_assets, 1.0 / n_assets)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != (n_assets,):
            raise ValueError(f"weights must have shape ({n_assets},), got {weight_array.shape}.")
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must contain only finite values.")

    s0 = spot[:, 0:1, :]
    log_spot = np.log(spot / s0)
    basket = np.tensordot(spot, weight_array, axes=([2], [0]))
    basket0 = basket[:, 0:1]
    counts = np.arange(1, n_points + 1)
    running_average = np.cumsum(basket, axis=1) / counts

    tau = np.broadcast_to(times / times[-1], (n_paths, n_points))
    channels = [
        tau,
        log_spot[:, :, 0],
        log_spot[:, :, 1],
        log_spot[:, :, 2],
    ]
    names = ["time", "log_price_1", "log_price_2", "log_price_3"]

    if vol is not None:
        vol = np.asarray(vol, dtype=float)
        if vol.shape != spot.shape:
            raise ValueError(f"vol must match spot shape {spot.shape}, got {vol.shape}.")
        for asset_index in range(n_assets):
            channels.append(vol[:, :, asset_index])
            names.append(f"{vol_name}_{asset_index + 1}")

    channels.append(np.log(running_average / basket0))
    names.append("running_basket_average")
    return PathFeatures(values=np.stack(channels, axis=-1), names=tuple(names))


def assemble_linear_hedge_features(
    features: PathFeatures,
    times: np.ndarray,
    increments: np.ndarray,
) -> np.ndarray:
    """Collapse each path into the hedge row ``Psi`` for a *linear* readout.

    Here the state is the current feature vector itself. Since feature builders
    already include a time channel, that channel is removed before
    :func:`readout_features` prepends ``[1, t_k]``. Thus a feature set with
    first channel ``time`` has per-step width ``d + 1`` rather than ``d + 2``.
    """
    current = _current_state_block(features)
    phi = readout_features(current, times)
    return control_features(phi, increments)


def _current_state_block(features: PathFeatures) -> np.ndarray:
    """Current feature channels used directly in nonlinear hedge readouts.

    The readout already prepends time as a separate column, so feature sets whose
    first channel is named ``time`` contribute only the remaining state channels.
    """
    if features.names and features.names[0] == "time":
        return features.values[:, :, 1:]
    return features.values


def _quadratic_state_block(features: PathFeatures) -> np.ndarray:
    """Current non-time state plus all degree-two monomials."""
    current = _current_state_block(features)
    blocks = [current]
    n_channels = current.shape[-1]
    for left in range(n_channels):
        for right in range(left, n_channels):
            blocks.append((current[:, :, left] * current[:, :, right])[:, :, None])
    return np.concatenate(blocks, axis=-1)


def assemble_quadratic_hedge_features(
    features: PathFeatures,
    times: np.ndarray,
    increments: np.ndarray,
) -> np.ndarray:
    """Collapse paths for a quadratic hedge readout on the current state."""
    phi = readout_features(_quadratic_state_block(features), times)
    return control_features(phi, increments)


def assemble_reservoir_hedge_features(
    reservoir: RandomizedSignatureReservoir,
    features: PathFeatures,
    times: np.ndarray,
    increments: np.ndarray,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Collapse paths for a reservoir hedge readout ``[1, t_k, x_k, h_k]``."""
    n_paths = features.values.shape[0]
    current_dim = _current_state_block(features).shape[-1]
    m = 1 if increments.ndim == 2 else increments.shape[2]
    q = (reservoir.dim + current_dim + 2) * m
    psi = np.empty((n_paths, q))
    for start in range(0, n_paths, batch_size):
        stop = min(start + batch_size, n_paths)
        batch = PathFeatures(features.values[start:stop], features.names)
        states = reservoir.run(batch)
        current = _current_state_block(batch)
        phi = readout_features(np.concatenate([current, states], axis=-1), times)
        psi[start:stop] = control_features(phi, increments[start:stop])
    return psi


def assemble_signature_hedge_features(
    features: PathFeatures,
    times: np.ndarray,
    increments: np.ndarray,
    level: int,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Collapse paths for a signature hedge readout ``[1, t_k, x_k, s_k]``."""
    feature_increments = features.increments
    n_paths, _, n_channels = feature_increments.shape
    current = _current_state_block(features)
    current_dim = current.shape[-1]
    m = 1 if increments.ndim == 2 else increments.shape[2]
    q_phi = 2 + current_dim + signature_dim(n_channels, level)
    q = q_phi * m
    psi = np.empty((n_paths, q))

    for start in range(0, n_paths, batch_size):
        stop = min(start + batch_size, n_paths)
        batch = stop - start
        row = np.zeros((batch, q))
        shocks = increments[start:stop]
        for step, signature_state in enumerate(
            _iterate_running_signature(feature_increments[start:stop], level)
        ):
            ones = np.ones((batch, 1))
            time_column = np.full((batch, 1), times[step])
            phi = np.concatenate(
                [ones, time_column, current[start:stop, step, :], signature_state],
                axis=1,
            )
            if shocks.ndim == 2:
                row += phi * shocks[:, step][:, None]
            else:
                row += (phi[:, None, :] * shocks[:, step, :][:, :, None]).reshape(batch, -1)
        psi[start:stop] = row
    return psi


def heston_variance_particle_filter(
    model,
    spot: np.ndarray,
    signal: np.ndarray,
    times: np.ndarray,
    eta: float,
    n_particles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal particle filter for ``E[V_k | S_0:k, Y_0:k]`` in one-asset Heston.

    The filter is model-aware and uses the known Heston parameters. It returns
    only filtered variance mean and variance; the hedge still trades only
    discounted stock increments.
    """
    spot = np.asarray(spot, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if spot.ndim != 2:
        raise ValueError(f"spot must have shape (n, N + 1), got {spot.shape}.")
    if signal.shape != spot.shape:
        raise ValueError(f"signal must match spot shape {spot.shape}, got {signal.shape}.")
    if times.shape[0] != spot.shape[1]:
        raise ValueError(f"times must have length {spot.shape[1]}, got {times.shape[0]}.")
    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}.")
    if n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got {n_particles}.")
    if eta == 0.0:
        return signal.copy(), np.zeros_like(signal)

    n_paths, n_points = spot.shape
    n_steps = n_points - 1
    particles = np.full((n_paths, n_particles), float(model.v0))
    filtered_mean = np.empty_like(signal)
    filtered_var = np.empty_like(signal)
    filtered_mean[:, 0] = model.v0
    filtered_var[:, 0] = 0.0

    log_spot = np.log(spot)
    obs_var = eta * eta
    log_obs_norm = np.log(2.0 * np.pi * obs_var)
    rho_perp = np.sqrt(max(1.0 - model.rho**2, 0.0))
    eps = 1e-12

    for step in range(n_steps):
        dt = float(times[step + 1] - times[step])
        log_return = log_spot[:, step + 1] - log_spot[:, step]
        v_prev = np.maximum(particles, 0.0)

        stock_var = np.maximum(v_prev * dt, eps)
        stock_mean = (model.r - 0.5 * v_prev) * dt
        stock_residual = log_return[:, None] - stock_mean
        logw = -0.5 * (stock_residual**2 / stock_var + np.log(2.0 * np.pi * stock_var))

        sqrt_v = np.sqrt(v_prev)
        implied_dw_spot = np.zeros_like(v_prev)
        active = sqrt_v > 1e-10
        implied_dw_spot[active] = stock_residual[active] / sqrt_v[active]
        dw_perp = np.sqrt(dt) * rng.standard_normal((n_paths, n_particles))
        dw_var = model.rho * implied_dw_spot + rho_perp * dw_perp
        proposal = (
            v_prev
            + model.kappa * (model.theta - v_prev) * dt
            + model.xi * sqrt_v * dw_var
        )
        particles_next = np.maximum(proposal, 0.0)

        obs_residual = signal[:, step + 1, None] - particles_next
        logw += -0.5 * (obs_residual**2 / obs_var + log_obs_norm)
        logw -= np.max(logw, axis=1, keepdims=True)
        weights = np.exp(logw)
        weights /= np.sum(weights, axis=1, keepdims=True)

        mean = np.sum(weights * particles_next, axis=1)
        second = np.sum(weights * particles_next**2, axis=1)
        filtered_mean[:, step + 1] = mean
        filtered_var[:, step + 1] = np.maximum(second - mean**2, 0.0)

        positions = (rng.random((n_paths, 1)) + np.arange(n_particles)[None, :]) / n_particles
        cumulative = np.cumsum(weights, axis=1)
        indices = np.empty((n_paths, n_particles), dtype=int)
        for path_index in range(n_paths):
            indices[path_index] = np.searchsorted(
                cumulative[path_index],
                positions[path_index],
                side="right",
            )
        indices = np.minimum(indices, n_particles - 1)
        particles = np.take_along_axis(particles_next, indices, axis=1)

    return filtered_mean, filtered_var


def particle_filter_hedging_features(
    spot: np.ndarray,
    times: np.ndarray,
    signal: np.ndarray,
    variance_mean: np.ndarray,
    variance_var: np.ndarray,
    variance_center: float,
) -> PathFeatures:
    """Build hedge features from raw signal plus filtered variance diagnostics."""
    signal = np.asarray(signal, dtype=float)
    if signal.shape != spot.shape:
        raise ValueError("signal must match spot shape.")
    base = hedging_features(spot, times, signal - variance_center, "signal")
    if variance_mean.shape != spot.shape or variance_var.shape != spot.shape:
        raise ValueError("filtered variance mean/variance must match spot shape.")
    values = np.concatenate(
        [
            base.values,
            (variance_mean - variance_center)[..., None],
            variance_var[..., None],
        ],
        axis=-1,
    )
    return PathFeatures(
        values=values,
        names=base.names
        + (
            "pf_variance_mean",
            "pf_variance_var",
        ),
    )


@dataclass(frozen=True)
class HedgeReport:
    """Out-of-sample hedging-error metrics on a test sample.

    The hedging error is ``e_i = H_i - c - sum_k theta_k dS~_{k+1}``. The headline
    metric is the hedge-error variance reduction ``Var(H) / Var(e)`` (analogous to
    the pricing VR); ``mean(e) ~ 0`` and ``capital ~ price`` are the consistency
    checks (a self-financing hedge started with the fair price should break even on
    average).
    """

    n_eval: int
    price: float                 # E[H] on the test sample
    capital: float               # fitted initial capital c (the intercept)
    payoff_std: float            # std(H)
    error_std: float             # std(e)
    rmse: float                  # sqrt(mean(e^2)), includes any residual mean bias
    error_mse: float             # mean(e^2), the quadratic hedging risk estimate
    error_var: float             # Var(e)
    variance_reduction: float    # Var(H) / Var(e)
    mean_error: float            # mean(e); should be ~ 0
    mean_error_se: float
    errors: np.ndarray | None = None

    def summary(self) -> str:
        """Human-readable multi-line report."""
        return "\n".join(
            [
                f"Hedging evaluation on {self.n_eval:,} independent test paths",
                f"    price E[H]              = {self.price:.4f}",
                f"    initial capital c       = {self.capital:.4f}",
                f"    unhedged std  std(H)    = {self.payoff_std:.4f}",
                f"    hedged   std  std(e)    = {self.error_std:.4f}",
                f"    RMSE                    = {self.rmse:.4f}",
                f"    MSE risk                = {self.error_mse:.4f}",
                f"    variance reduction      = Var(H)/Var(e) = {self.variance_reduction:.2f}x",
                "",
                "    break-even check (mean hedging error should be ~ 0):",
                f"        mean(e) = {self.mean_error:+.4f} +/- {1.96 * self.mean_error_se:.4f} (95% CI)",
            ]
        )


def evaluate_hedge(
    control: RidgeMartingaleControl,
    psi: np.ndarray,
    payoff: np.ndarray,
) -> HedgeReport:
    """Score a fitted hedge on test paths.

    ``control.control(psi)`` is the discounted trading gain ``sum_k theta_k dS~``
    and ``control.intercept`` is the initial capital ``c``, so the hedging error is
    ``e = H - c - gain``.
    """
    gain = control.control(psi)
    error = payoff - control.intercept - gain
    n = int(payoff.size)

    error_var = float(error.var(ddof=1))
    payoff_var = float(payoff.var(ddof=1))
    return HedgeReport(
        n_eval=n,
        price=float(payoff.mean()),
        capital=float(control.intercept),
        payoff_std=float(np.sqrt(payoff_var)),
        error_std=float(np.sqrt(error_var)),
        rmse=float(np.sqrt(np.mean(error**2))),
        error_mse=float(np.mean(error**2)),
        error_var=error_var,
        variance_reduction=payoff_var / error_var if error_var > 0 else np.inf,
        mean_error=float(error.mean()),
        mean_error_se=float(error.std(ddof=1) / np.sqrt(n)),
        errors=error,
    )


def fit_hedge(
    psi_train: np.ndarray,
    payoff_train: np.ndarray,
    psi_val: np.ndarray,
    payoff_val: np.ndarray,
    lambdas: tuple[float, ...],
) -> tuple[RidgeMartingaleControl, float]:
    """Tune the ridge penalty on validation and fit the hedge readout on train.

    The validation criterion is the actual quadratic hedging objective
    ``mean((H - c - gain)^2)`` with the fitted intercept kept as the initial
    capital. This matches the observed-filtration GKW projection objective used
    in the proposal.
    """
    best_lambda = lambdas[0]
    best_mse = np.inf
    for lam in lambdas:
        candidate = RidgeMartingaleControl.fit(psi_train, payoff_train, lam)
        error = payoff_val - candidate.intercept - candidate.control(psi_val)
        mse = float(np.mean(error**2))
        if mse < best_mse:
            best_lambda = lam
            best_mse = mse
    control = RidgeMartingaleControl.fit(psi_train, payoff_train, best_lambda)
    return control, best_lambda

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

    Psi_i = sum_{k=0}^{N-1} phi_{i,k} * dS~_{i,k+1},   phi_k = [1, t_k, h_k],

the best linear hedge is exactly the ridge problem

    min_{c, beta}  sum_i ( H_i - c - beta . Psi_i )^2 + lambda ||beta||^2,

i.e. the pricing control-variate regression with the simulator increment ``dM``
replaced by the traded increment ``dS~`` and the intercept *kept* as the initial
capital ``c``. We therefore reuse :class:`RidgeMartingaleControl`,
:func:`assemble_control_features` and :func:`assemble_signature_control_features`
unchanged; only the increment, the target and the reporting differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import RidgeMartingaleControl, control_features
from .features import PathFeatures
from .reservoir import readout_features


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

    Here the state is the feature vector itself, ``theta_k = beta . [1, t_k, x_k]``
    (an "identity reservoir"). This is the baseline used by the explicit-feature
    hedges (oracle / raw signal). It reuses :func:`readout_features` to
    prepend ``[1, t_k]`` and :func:`control_features` to form
    ``Psi_i = sum_k phi_{i,k} dS~_{i,k+1}``, so ``q = d + 2``.
    """
    phi = readout_features(features.values, times)     # (n, N + 1, d + 2)
    return control_features(phi, increments)


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

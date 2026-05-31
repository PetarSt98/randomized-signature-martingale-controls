"""Non-anticipative preprocessing: raw price paths -> feature channels x_k.

Appendix construction
---------------------
    x_k = Psi_k(X_{0:k}) in R^d,        F_{t_k}-measurable,
    Delta x_{k+1} = x_{k+1} - x_k.

The reservoir is *driven by the increments* ``Delta x``, so the choice of
channels decides what path information the reservoir can see.

Why three channels (not one)
----------------------------
The raw state is the scalar price S_k (d = 1). A signature / randomized
signature of a 1-D path is degenerate: it only depends on the net increment
S_T - S_0, so it cannot see path ordering or averages. We therefore use an
Asian-aware, time-augmented set of channels:

    1. time           tau_k    = t_k / T              (time augmentation)
    2. log-price      log(S_k / S_0)                  (current price level)
    3. running average log(A_k / S_0), A_k = mean(S_0..S_k)   (Asian state)

These three are the *sufficient* state for the arithmetic Asian call. Two more
genuine path-state channels are available (off by default) for richer exotics
and for studying how the signature dimension scales with d:

    4. running max    max_{j<=k} log(S_j / S_0)       (lookback / barrier state)
    5. running min    min_{j<=k} log(S_j / S_0)       (lookback / barrier state)

All channels start at 0 at k = 0, which matches the reservoir's initial state
h_0 = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import SimulatedPaths
from .commodity import CommodityForwardCurvePaths
from .heston import HestonPaths
from .multi_asset_heston import MultiAssetHestonPaths


@dataclass(frozen=True)
class PathFeatures:
    """Preprocessed feature paths and their channel names.

    Attributes
    ----------
    values:
        Feature paths x_0, ..., x_N, shape ``(n_paths, N + 1, d)``.
    names:
        Human-readable name of each of the ``d`` channels.
    """

    values: np.ndarray
    names: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        """Number of feature channels d."""
        return self.values.shape[-1]

    @property
    def increments(self) -> np.ndarray:
        """Increments Delta x_{k+1} = x_{k+1} - x_k, shape ``(n_paths, N, d)``."""
        return np.diff(self.values, axis=1)


def asian_features(
    paths: SimulatedPaths,
    include_time: bool = True,
    include_log_price: bool = True,
    include_running_average: bool = True,
    include_running_max: bool = False,
    include_running_min: bool = False,
) -> PathFeatures:
    """Build Asian-aware, time-augmented feature channels from price paths.

    By default returns the three channels (time, log-price, running average) -
    the sufficient state for the arithmetic Asian call. Turning on the running
    max / min of log-price adds two more genuine path-state channels (d = 5),
    useful for richer exotics and for studying signature-dimension scaling. Each
    flag can be toggled; at least one channel must remain.

    Returns
    -------
    PathFeatures
        Feature paths of shape ``(n_paths, N + 1, d)`` and the channel names.
    """
    spot = paths.spot                      # (n, N + 1)
    s0 = spot[:, 0:1]                       # (n, 1), the initial price per path
    n_paths, n_points = spot.shape
    log_price = np.log(spot / s0)          # 0 at k = 0; shared by several channels

    channels: list[np.ndarray] = []
    names: list[str] = []

    if include_time:
        maturity = paths.times[-1]
        tau = np.broadcast_to(paths.times / maturity, (n_paths, n_points))
        channels.append(tau)
        names.append("time")

    if include_log_price:
        channels.append(log_price)
        names.append("log_price")

    if include_running_average:
        running_sum = np.cumsum(spot, axis=1)
        counts = np.arange(1, n_points + 1)            # [1, 2, ..., N + 1]
        running_average = running_sum / counts         # A_k = mean(S_0..S_k)
        channels.append(np.log(running_average / s0))
        names.append("running_average")

    if include_running_max:                            # running maximum of log-price
        channels.append(np.maximum.accumulate(log_price, axis=1))
        names.append("running_max")

    if include_running_min:                            # running minimum of log-price
        channels.append(np.minimum.accumulate(log_price, axis=1))
        names.append("running_min")

    if not channels:
        raise ValueError("At least one feature channel must be enabled.")

    values = np.stack(channels, axis=-1)               # (n, N + 1, d)
    return PathFeatures(values=values, names=tuple(names))


def heston_features(
    paths: HestonPaths,
    include_time: bool = True,
    include_log_price: bool = True,
    include_variance: bool = True,
    include_running_average: bool = True,
    include_running_max: bool = False,
    include_running_min: bool = False,
) -> PathFeatures:
    """Adapted feature channels for Heston paths (the stochastic-variance state).

    Mirrors :func:`asian_features` but adds the **variance** as a genuine state
    channel (Heston's second factor). All channels are F_{t_k}-measurable - no
    future information - and start at 0 at k = 0 (matching the reservoir's
    h_0 = 0). The default four channels are

        1. time            tau_k = t_k / T
        2. log-price       log(S_k / S_0)
        3. variance        v_k - v_0           (centred to start at 0)
        4. running average log(A_k / S_0), A_k = mean(S_0..S_k)

    with optional running max / min of log-price for lookback / barrier state.
    The variance is centred at v_0 only for a clean starting value; the
    reservoir and signature are driven by *increments*, which are unaffected by
    this constant shift.

    Returns
    -------
    PathFeatures
        Feature paths of shape ``(n_paths, N + 1, d)`` and the channel names.
    """
    spot = paths.spot                      # (n, N + 1)
    variance = paths.variance              # (n, N + 1)
    s0 = spot[:, 0:1]                       # (n, 1)
    v0 = variance[:, 0:1]                   # (n, 1)
    n_paths, n_points = spot.shape
    log_price = np.log(spot / s0)          # 0 at k = 0

    channels: list[np.ndarray] = []
    names: list[str] = []

    if include_time:
        maturity = paths.times[-1]
        tau = np.broadcast_to(paths.times / maturity, (n_paths, n_points))
        channels.append(tau)
        names.append("time")

    if include_log_price:
        channels.append(log_price)
        names.append("log_price")

    if include_variance:
        channels.append(variance - v0)                 # v_k - v_0, starts at 0
        names.append("variance")

    if include_running_average:
        running_sum = np.cumsum(spot, axis=1)
        counts = np.arange(1, n_points + 1)
        running_average = running_sum / counts          # A_k = mean(S_0..S_k)
        channels.append(np.log(running_average / s0))
        names.append("running_average")

    if include_running_max:
        channels.append(np.maximum.accumulate(log_price, axis=1))
        names.append("running_max")

    if include_running_min:
        channels.append(np.minimum.accumulate(log_price, axis=1))
        names.append("running_min")

    if not channels:
        raise ValueError("At least one feature channel must be enabled.")

    values = np.stack(channels, axis=-1)               # (n, N + 1, d)
    return PathFeatures(values=values, names=tuple(names))


def multi_asset_heston_features(
    paths: MultiAssetHestonPaths,
    weights: np.ndarray | tuple[float, ...] | None = None,
) -> PathFeatures:
    """Feature map for a three-asset Heston basket Asian call, exactly ``d = 8``.

    Channels are time, three log-spot ratios, three centred variances, and the
    running basket-average log ratio. Everything is adapted and starts at zero.
    """
    spot = paths.spot                                  # (n, N + 1, 3)
    variance = paths.variance                          # (n, N + 1, 3)
    if spot.ndim != 3 or spot.shape[2] != 3:
        raise ValueError(f"spot must have shape (n_paths, N + 1, 3), got {spot.shape}.")
    if variance.shape != spot.shape:
        raise ValueError(f"variance must match spot shape {spot.shape}, got {variance.shape}.")

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
    v0 = variance[:, 0:1, :]
    log_spot = np.log(spot / s0)
    basket = np.tensordot(spot, weight_array, axes=([2], [0]))     # (n, N + 1)
    basket0 = basket[:, 0:1]
    running_sum = np.cumsum(basket, axis=1)
    counts = np.arange(1, n_points + 1)
    running_average = running_sum / counts

    maturity = paths.times[-1]
    tau = np.broadcast_to(paths.times / maturity, (n_paths, n_points))
    channels = [
        tau,
        log_spot[:, :, 0],
        log_spot[:, :, 1],
        log_spot[:, :, 2],
        variance[:, :, 0] - v0[:, :, 0],
        variance[:, :, 1] - v0[:, :, 1],
        variance[:, :, 2] - v0[:, :, 2],
        np.log(running_average / basket0),
    ]
    names = (
        "time",
        "log_price_1",
        "log_price_2",
        "log_price_3",
        "variance_1",
        "variance_2",
        "variance_3",
        "running_basket_average",
    )
    return PathFeatures(values=np.stack(channels, axis=-1), names=names)


def commodity_crack_spread_features(
    paths: CommodityForwardCurvePaths,
    average_node: int = 0,
) -> PathFeatures:
    """Feature map for the commodity crack-spread benchmark, exactly ``d = 12``.

    The first three channels are running averages for crude, gasoline and
    distillate on the selected averaging node. The remaining nine channels are
    log-forward ratios for the full 3-by-3 curve. Time is intentionally not a
    channel here because the control readout already receives ``t_k``.
    """
    curves = paths.forward_curves                      # (n, N + 1, 3, 3)
    if curves.ndim != 4 or curves.shape[2:] != (3, 3):
        raise ValueError(
            "forward_curves must have shape (n_paths, N + 1, 3, 3), "
            f"got {curves.shape}."
        )
    if not 0 <= average_node < curves.shape[3]:
        raise ValueError(f"average_node must be in [0, {curves.shape[3] - 1}].")

    n_paths, n_points, _, _ = curves.shape
    selected = curves[:, :, :, average_node]           # (n, N + 1, 3)
    running_sum = np.cumsum(selected, axis=1)
    counts = np.arange(1, n_points + 1)
    running_average = running_sum / counts[:, None]
    running_average0 = running_average[:, 0:1, :]
    log_running_average = np.log(running_average / running_average0)

    initial_curves = curves[:, 0:1, :, :]
    log_forwards = np.log(curves / initial_curves)

    channels: list[np.ndarray] = [
        log_running_average[:, :, 0],
        log_running_average[:, :, 1],
        log_running_average[:, :, 2],
    ]
    names = [
        "running_avg_crude",
        "running_avg_gasoline",
        "running_avg_distillate",
    ]
    for commodity_index, commodity in enumerate(paths.commodity_names):
        for node_index, node in enumerate(paths.node_names):
            channels.append(log_forwards[:, :, commodity_index, node_index])
            names.append(f"log_{commodity}_{node}")

    values = np.stack(channels, axis=-1)
    if values.shape != (n_paths, n_points, 12):
        raise RuntimeError(f"commodity features should have d = 12, got {values.shape}.")
    return PathFeatures(values=values, names=tuple(names))

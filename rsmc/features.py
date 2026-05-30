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
from .heston import HestonPaths


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

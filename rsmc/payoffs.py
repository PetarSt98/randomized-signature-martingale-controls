"""Discounted payoffs and a closed-form Black-Scholes price for validation.

Every function here returns the *discounted* payoff G_N = e^{-rT} * payoff,
which is the Monte Carlo target p_N = E_Q[G_N] in the proposal. Returning
discounted values keeps the rest of the pipeline simple: the control variate
acts directly on G_N.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.stats import norm

from .black_scholes import SimulatedPaths


class PricePaths(Protocol):
    """Anything carrying spot prices and a time grid (Black-Scholes or Heston).

    The path-dependent payoffs only need the prices and the time grid, so they
    work unchanged on any simulator output that exposes these two arrays.
    """

    spot: np.ndarray
    times: np.ndarray


def arithmetic_asian_call(
    paths: PricePaths,
    strike: float,
    rate: float,
    include_initial: bool = False,
) -> np.ndarray:
    """Discounted arithmetic Asian call payoff, one value per path.

    The arithmetic average is taken over the monitoring dates and the payoff is

        G_N = e^{-rT} * max(A - K, 0),     A = average of S over the grid.

    Parameters
    ----------
    paths:
        Simulated paths (provides the prices and the time grid).
    strike:
        Strike price K.
    rate:
        Risk-free rate r used for discounting over [0, T].
    include_initial:
        If ``True`` the known initial price S_0 is part of the average;
        otherwise the average runs over the post-initial dates S_1, ..., S_N
        (the usual convention for a discretely monitored Asian option).

    Returns
    -------
    np.ndarray
        Discounted payoffs, shape ``(n_paths,)``.
    """
    monitored = paths.spot if include_initial else paths.spot[:, 1:]
    average = monitored.mean(axis=1)
    maturity = paths.times[-1]
    discount = np.exp(-rate * maturity)
    return discount * np.maximum(average - strike, 0.0)


def european_call(
    paths: SimulatedPaths,
    strike: float,
    rate: float,
) -> np.ndarray:
    """Discounted European call payoff G_N = e^{-rT} * max(S_N - K, 0).

    Used as a sanity check: its Monte Carlo mean must match
    :func:`black_scholes_call_price`.

    Returns
    -------
    np.ndarray
        Discounted payoffs, shape ``(n_paths,)``.
    """
    terminal = paths.spot[:, -1]
    maturity = paths.times[-1]
    discount = np.exp(-rate * maturity)
    return discount * np.maximum(terminal - strike, 0.0)


def black_scholes_call_price(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
) -> float:
    """Closed-form Black-Scholes price of a European call.

        C = S_0 * Phi(d1) - K * e^{-rT} * Phi(d2),
        d1 = (log(S_0 / K) + (r + sigma^2 / 2) T) / (sigma sqrt(T)),
        d2 = d1 - sigma sqrt(T),

    where Phi is the standard normal CDF. This is the ground truth the Monte
    Carlo European-call estimate should converge to.
    """
    sqrt_t = sigma * np.sqrt(maturity)
    d1 = (np.log(s0 / strike) + (rate + 0.5 * sigma**2) * maturity) / sqrt_t
    d2 = d1 - sqrt_t
    return float(s0 * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def geometric_asian_call_price(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_steps: int,
) -> float:
    """Closed-form price of a discretely monitored *geometric* Asian call.

    The geometric average over the monitoring dates t_1, ..., t_N is lognormal
    under Black-Scholes, so the price has a Black-Scholes-like closed form. This
    is used only as a cheap, exact reference: by the AM-GM inequality the
    arithmetic Asian call (our target) is priced slightly *above* this value.

    The averaging window (t_1, ..., t_N) matches the default of
    :func:`arithmetic_asian_call`.
    """
    dt = maturity / n_steps
    mean_time = dt * (n_steps + 1) / 2                    # average of t_1, ..., t_N

    # log of the geometric average is normal with this mean and variance.
    mean_log = np.log(s0) + (rate - 0.5 * sigma**2) * mean_time
    var_log = sigma**2 * dt * (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps)
    std_log = np.sqrt(var_log)

    d1 = (mean_log + var_log - np.log(strike)) / std_log
    d2 = d1 - std_log
    forward = np.exp(mean_log + 0.5 * var_log)            # E[geometric average]
    return float(np.exp(-rate * maturity) * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))

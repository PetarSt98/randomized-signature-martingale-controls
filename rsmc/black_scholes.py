"""Black-Scholes path simulator under the risk-neutral measure Q.

This module generates Monte Carlo paths for the Black-Scholes model. It is the
first building block of the project: later steps (reservoirs, signatures,
martingale-control variates) all consume the paths produced here.

Model
-----
Under the pricing measure Q the asset price follows

    dS_t = r S_t dt + sigma S_t dW_t,

where r is the risk-free rate, sigma the volatility and W a Q-Brownian motion.
On a grid 0 = t_0 < t_1 < ... < t_N = T with step dt = T / N this SDE has an
*exact* solution (no discretisation error):

    S_{k+1} = S_k * exp( (r - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z_{k+1} ),

with i.i.d. standard normal innovations Z_{k+1} ~ N(0, 1).

Why we keep the innovations
---------------------------
The project builds a martingale control variate

    C(Z) = sum_{k=0}^{N-1} Z_k . dM_{k+1},          E_Q[dM_{k+1} | F_{t_k}] = 0.

In Black-Scholes the martingale increments dM_{k+1} are exactly these Brownian
innovations Z_{k+1} (they have conditional mean zero by construction). So the
simulator returns the innovations alongside the price paths: the prices feed
the payoff G_N, the innovations feed the control variate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlackScholesModel:
    """Parameters of a Black-Scholes model under the risk-neutral measure Q.

    Attributes
    ----------
    s0:
        Initial asset price S_0 (> 0).
    r:
        Continuously compounded risk-free rate.
    sigma:
        Volatility (> 0).
    T:
        Maturity in years (> 0).
    """

    s0: float = 100.0
    r: float = 0.03
    sigma: float = 0.20
    T: float = 1.0

    def __post_init__(self) -> None:
        if self.s0 <= 0:
            raise ValueError(f"s0 must be positive, got {self.s0}.")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> "SimulatedPaths":
        """Simulate exact Black-Scholes paths on a uniform time grid.

        Parameters
        ----------
        n_paths:
            Number of independent paths to simulate.
        n_steps:
            Number of time steps N (the grid then has N + 1 points).
        rng:
            A NumPy random generator, e.g. ``np.random.default_rng(0)``.
            Passing it explicitly keeps simulations reproducible and lets us
            draw independent train / validation / test samples.

        Returns
        -------
        SimulatedPaths
            Bundle of the time grid, the price paths and the Brownian
            innovations (the martingale increments).
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}.")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}.")

        dt = self.T / n_steps
        times = np.linspace(0.0, self.T, n_steps + 1)

        # Standard normal innovations Z_{k+1}, one per step and per path.
        innovations = rng.standard_normal(size=(n_paths, n_steps))

        # Exact log-increments  log S_{k+1} - log S_k.
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * innovations
        log_increments = drift + diffusion

        # Accumulate log-increments, then prepend log(S_0) and exponentiate.
        log_spot = np.empty((n_paths, n_steps + 1))
        log_spot[:, 0] = np.log(self.s0)
        log_spot[:, 1:] = np.log(self.s0) + np.cumsum(log_increments, axis=1)
        spot = np.exp(log_spot)

        return SimulatedPaths(times=times, spot=spot, innovations=innovations)


@dataclass(frozen=True)
class SimulatedPaths:
    """A batch of simulated paths and the innovations that generated them.

    Shapes use ``n`` for the number of paths and ``N`` for the number of steps.

    Attributes
    ----------
    times:
        Time grid (t_0, ..., t_N), shape ``(N + 1,)``.
    spot:
        Asset prices (S_0, ..., S_N), shape ``(n, N + 1)``.
    innovations:
        Standard normal Brownian innovations Z_{k+1}, shape ``(n, N)``.
        These are the martingale increments dM_{k+1} used by the control
        variate: E_Q[Z_{k+1} | F_{t_k}] = 0 and Var_Q[Z_{k+1} | F_{t_k}] = 1.
    """

    times: np.ndarray
    spot: np.ndarray
    innovations: np.ndarray

    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of time steps N."""
        return self.spot.shape[1] - 1

    @property
    def dt(self) -> float:
        """Uniform time step dt = T / N."""
        return float(self.times[1] - self.times[0])

    @property
    def brownian_increments(self) -> np.ndarray:
        """Brownian increments dW_{k+1} = sqrt(dt) * Z_{k+1}, shape ``(n, N)``.

        Provided for convenience: the un-normalised increments are sometimes
        more natural to read. The control variate is unaffected by the choice
        between Z and dW because the constant sqrt(dt) is absorbed by the
        trained readout.
        """
        return np.sqrt(self.dt) * self.innovations

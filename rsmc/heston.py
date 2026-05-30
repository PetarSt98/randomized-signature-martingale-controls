"""Heston path simulator under the risk-neutral measure Q (full-truncation Euler).

Model
-----
Under the pricing measure Q the Heston model is the two-factor SDE

    dS_t = r S_t dt + sqrt(v_t) S_t dW^S_t,
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,
    d<W^S, W^v>_t = rho dt,

with a stochastic variance v_t (a CIR process) correlated with the spot
Brownian motion. There are now *two* Brownian drivers, so the martingale control
variate uses a **vector** increment dM_{k+1} = (dW^S_{k+1}, dW^v_{k+1}) in R^2.

Discretisation
--------------
Unlike Black-Scholes, Heston has no exact simulation on a coarse grid. We use a
**full-truncation Euler** scheme: the variance is updated by Euler and truncated
at 0 each step, so it stays nonnegative and the square roots are always well
defined,

    v_{k+1}     = max( v_k + kappa (theta - v_k) dt + xi sqrt(v_k) dW^v_{k+1}, 0 ),
    log S_{k+1} = log S_k + (r - 0.5 v_k) dt + sqrt(v_k) dW^S_{k+1}.

The log-Euler step for S uses v_k consistently in its drift and diffusion, so the
discounted spot stays a martingale in expectation (E[S_{k+1} | F_k] = S_k e^{r dt})
and S never goes negative.

IMPORTANT: this prices the *discretised* Heston simulation target - the law of
the Euler scheme on this grid - NOT the exact continuous-time Heston price.
Refining ``n_steps`` drives the discrete target towards the continuous one.

Why we keep the Brownian increments
-----------------------------------
The same correlated increments (dW^S, dW^v) that generated the path are returned
and fed - unchanged - to the control variate. Each is a mean-zero martingale
increment given F_{t_k}, so C = sum_k Z_k . dM_{k+1} is an exact zero-mean
martingale (unbiased), exactly as in the Black-Scholes case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonModel:
    """Parameters of a Heston model under the risk-neutral measure Q.

    Attributes
    ----------
    s0:
        Initial asset price S_0 (> 0).
    v0:
        Initial variance v_0 (>= 0).
    r:
        Continuously compounded risk-free rate.
    kappa:
        Mean-reversion speed of the variance (> 0).
    theta:
        Long-run variance level (> 0).
    xi:
        Volatility of variance ("vol of vol", > 0).
    rho:
        Correlation between the spot and variance Brownian motions (in [-1, 1]).
    T:
        Maturity in years (> 0).
    """

    s0: float = 100.0
    v0: float = 0.04
    r: float = 0.03
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.30
    rho: float = -0.70
    T: float = 1.0

    def __post_init__(self) -> None:
        if self.s0 <= 0:
            raise ValueError(f"s0 must be positive, got {self.s0}.")
        if self.v0 < 0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}.")
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            raise ValueError("kappa, theta and xi must be positive.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> "HestonPaths":
        """Simulate Heston paths with the full-truncation Euler scheme.

        Parameters
        ----------
        n_paths:
            Number of independent paths to simulate.
        n_steps:
            Number of time steps N (the grid then has N + 1 points).
        rng:
            A NumPy random generator (passed explicitly for reproducible,
            independent train / validation / test samples).

        Returns
        -------
        HestonPaths
            Time grid, spot paths, variance paths and the correlated Brownian
            increments (dW^S, dW^v) of shape ``(n, N, 2)`` that drove them.
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}.")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}.")

        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, self.T, n_steps + 1)

        # Two correlated standard normals per step: z_v = rho z_s + sqrt(1-rho^2) z_perp.
        z_spot = rng.standard_normal((n_paths, n_steps))
        z_perp = rng.standard_normal((n_paths, n_steps))
        z_var = self.rho * z_spot + np.sqrt(1.0 - self.rho**2) * z_perp

        # Brownian increments dW = sqrt(dt) * z; these are the martingale increments.
        brownian_increments = np.stack([z_spot, z_var], axis=2) * sqrt_dt   # (n, N, 2)

        log_spot = np.empty((n_paths, n_steps + 1))
        variance = np.empty((n_paths, n_steps + 1))
        log_spot[:, 0] = np.log(self.s0)
        variance[:, 0] = self.v0

        for k in range(n_steps):
            v_k = variance[:, k]                            # nonnegative (truncated below)
            sqrt_v = np.sqrt(v_k)
            dw_spot = brownian_increments[:, k, 0]
            dw_var = brownian_increments[:, k, 1]
            log_spot[:, k + 1] = log_spot[:, k] + (self.r - 0.5 * v_k) * dt + sqrt_v * dw_spot
            next_var = v_k + self.kappa * (self.theta - v_k) * dt + self.xi * sqrt_v * dw_var
            variance[:, k + 1] = np.maximum(next_var, 0.0)  # full truncation: keep variance >= 0

        return HestonPaths(
            times=times,
            spot=np.exp(log_spot),
            variance=variance,
            brownian_increments=brownian_increments,
        )


@dataclass(frozen=True)
class HestonPaths:
    """A batch of Heston paths and the correlated Brownian increments that drove them.

    Shapes use ``n`` for the number of paths and ``N`` for the number of steps.

    Attributes
    ----------
    times:
        Time grid (t_0, ..., t_N), shape ``(N + 1,)``.
    spot:
        Asset prices (S_0, ..., S_N), shape ``(n, N + 1)``.
    variance:
        Variance paths (v_0, ..., v_N), shape ``(n, N + 1)``, truncated at 0
        each step so they stay nonnegative.
    brownian_increments:
        Correlated Brownian increments (dW^S_{k+1}, dW^v_{k+1}), shape
        ``(n, N, 2)``. These are the vector martingale increments dM_{k+1} used
        by the control variate: E_Q[dM_{k+1} | F_{t_k}] = 0 in each component.
    """

    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
    brownian_increments: np.ndarray

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


def make_heston_datasets(
    model: HestonModel,
    sizes: dict[str, int],
    n_steps: int,
    seed: int = 0,
) -> dict[str, HestonPaths]:
    """Generate independent Heston datasets (e.g. train / validation / test).

    Mirrors :func:`rsmc.config.make_datasets`: a single master ``seed`` is split
    into independent sub-streams (one per requested size) so the splits are
    independent - which keeps the sample-split control estimator unbiased - and
    the whole experiment is reproducible.
    """
    names = list(sizes)
    sub_seeds = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: model.simulate(sizes[name], n_steps, np.random.default_rng(sub_seed))
        for name, sub_seed in zip(names, sub_seeds)
    }

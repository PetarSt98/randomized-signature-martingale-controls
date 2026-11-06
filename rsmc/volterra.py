"""Bounded Volterra volatility model on a fixed grid (report Table 1 / A.6).

Volatility is a bounded sigmoid of a Gaussian Volterra driver,

    sigma_k = sigma_min + sigma_range * sgm(X_k),      sgm(z) = 1 / (1 + exp(-z)),
    X_k     = sum_{j<k} g_{k-j} Delta W^v_{j+1},         X_0 = 0,

with cell-averaged Riemann-Liouville weights

    g_m = sqrt(2H) Delta^{H-1/2} / (H + 1/2) * [m^{H+1/2} - (m-1)^{H+1/2}],   m >= 1.

At Hurst index H = 1/2 every weight equals one, X is a random walk and
(t, S, A, sigma) is Markov. At H < 1/2 the weights decay like a power law, so
the whole driver history predicts future volatility beyond the current level.
The stock follows a log-Euler step with the current volatility,

    log S_{k+1} = log S_k + (r - sigma_k^2 / 2) Delta + sigma_k Delta B_{k+1},
    Delta B = rho Delta W^v + sqrt(1 - rho^2) Delta W^perp,

so the discounted price is an exact martingale on the grid and bounded
volatility gives finite moments of all orders. This is a bounded variant, not
rough Bergomi, and it is the grid simulator itself, not a discretization of a
continuous model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundedVolterraModel:
    s0: float = 100.0
    r: float = 0.03
    T: float = 1.0
    hurst: float = 0.1
    rho: float = -0.5
    sigma_min: float = 0.05
    sigma_range: float = 0.40

    def __post_init__(self) -> None:
        values = [self.s0, self.r, self.T, self.hurst, self.rho, self.sigma_min, self.sigma_range]
        if not np.isfinite(values).all():
            raise ValueError("All model parameters must be finite.")
        if self.s0 <= 0 or self.T <= 0:
            raise ValueError("s0 and T must be positive.")
        if not 0 < self.hurst <= 0.5:
            raise ValueError("hurst must lie in (0, 1/2].")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1].")
        if self.sigma_min <= 0 or self.sigma_range < 0:
            raise ValueError("sigma_min must be positive and sigma_range non-negative.")

    @property
    def sigma_initial(self) -> float:
        """sigma_0 = sigma_min + sigma_range / 2, since X_0 = 0."""
        return self.sigma_min + 0.5 * self.sigma_range

    def kernel_weights(self, n_steps: int) -> np.ndarray:
        """Weights g_1, ..., g_N; index 0 is unused and set to zero."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        dt = self.T / n_steps
        h = self.hurst
        m = np.arange(n_steps + 1, dtype=float)
        weights = np.zeros(n_steps + 1)
        weights[1:] = (np.sqrt(2.0 * h) * dt ** (h - 0.5) / (h + 0.5)
                       * (m[1:] ** (h + 0.5) - (m[1:] - 1.0) ** (h + 0.5)))
        return weights

    def driver(self, vol_increments: np.ndarray) -> np.ndarray:
        """Causal Volterra driver X_0..X_N from Delta W^v, shape (n, N) -> (n, N+1)."""
        vol_increments = np.asarray(vol_increments, dtype=float)
        if vol_increments.ndim != 2:
            raise ValueError("vol_increments must have shape (n_paths, N).")
        n_paths, n_steps = vol_increments.shape
        g = self.kernel_weights(n_steps)
        x = np.zeros((n_paths, n_steps + 1))
        for k in range(1, n_steps + 1):
            # X_k = sum_{j=0}^{k-1} g_{k-j} dW_j  ->  weights g_k, ..., g_1.
            x[:, k] = vol_increments[:, :k] @ g[k:0:-1]
        return x

    def volatility(self, driver: np.ndarray) -> np.ndarray:
        return self.sigma_min + self.sigma_range / (1.0 + np.exp(-driver))

    def simulate(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> "VolterraPaths":
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive.")
        dt = self.T / n_steps
        times = np.linspace(0.0, self.T, n_steps + 1)
        z_vol = rng.standard_normal((n_paths, n_steps))
        z_perp = rng.standard_normal((n_paths, n_steps))
        d_wv = np.sqrt(dt) * z_vol
        d_b = np.sqrt(dt) * (self.rho * z_vol + np.sqrt(1.0 - self.rho**2) * z_perp)
        driver = self.driver(d_wv)
        sigma = self.volatility(driver)
        # Only sigma_0..sigma_{N-1} drive the stock; sigma_N is observed at maturity.
        log_increments = (self.r - 0.5 * sigma[:, :-1] ** 2) * dt + sigma[:, :-1] * d_b
        log_spot = np.empty((n_paths, n_steps + 1))
        log_spot[:, 0] = np.log(self.s0)
        log_spot[:, 1:] = np.log(self.s0) + np.cumsum(log_increments, axis=1)
        increments = np.stack([d_b, d_wv], axis=-1)
        return VolterraPaths(times, np.exp(log_spot), sigma, driver, increments)


@dataclass(frozen=True)
class VolterraPaths:
    """Spot (n,N+1), volatility (n,N+1), driver X (n,N+1), increments (dB, dW^v) (n,N,2)."""

    times: np.ndarray
    spot: np.ndarray
    volatility: np.ndarray
    driver: np.ndarray
    brownian_increments: np.ndarray

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])


def make_volterra_datasets(model, sizes, n_steps, seed=0):
    """Independent child streams per split, as in the other simulators."""
    children = np.random.SeedSequence(seed).spawn(len(sizes))
    return {name: model.simulate(size, n_steps, np.random.default_rng(child))
            for (name, size), child in zip(sizes.items(), children)}

"""One stock with two independent Heston variance factors on a fixed grid.

Each variance uses projected Euler; the stock log step uses current variances
in both drift and diffusion. Brownian pairs (B_i, W_i) have correlation rho_i
within a pair and are independent across factors. This finite-factor numerical
target is not rough volatility or an exact continuous-time Heston simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoFactorHestonModel:
    s0: float = 100.0
    v0: tuple[float, float] = (0.02, 0.02)
    r: float = 0.03
    kappa: tuple[float, float] = (0.5, 5.0)
    theta: tuple[float, float] = (0.02, 0.02)
    xi: tuple[float, float] = (0.30, 0.30)
    rho: tuple[float, float] = (-0.70, -0.70)
    T: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.s0, self.r, self.T]).all() or self.s0 <= 0 or self.T <= 0:
            raise ValueError("s0 and T must be positive; scalar parameters must be finite.")
        for name in ("v0", "kappa", "theta", "xi", "rho"):
            values = np.asarray(getattr(self, name))
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain two finite values.")
            if name == "rho":
                valid = np.abs(values) <= 1
            else:
                valid = values >= 0 if name == "v0" else values > 0
            if not valid.all():
                raise ValueError(f"Invalid {name} parameters.")

    def step(self, log_spot, variance, dt, standard_normals):
        """Advance supplied states with normals [..., factor, (B, independent)].

        Return log spot, projected variance and increments (B1,W1,B2,W2).
        All terms in the stock step use the supplied current variance.
        """
        log_spot = np.asarray(log_spot, dtype=float)
        variance = np.asarray(variance, dtype=float)
        standard_normals = np.asarray(standard_normals, dtype=float)
        if (variance.shape != log_spot.shape + (2,)
                or standard_normals.shape != variance.shape + (2,)):
            raise ValueError("Expected two variance factors and two normals per factor.")
        if not np.isfinite(dt) or dt <= 0 or np.any(variance < 0):
            raise ValueError("dt must be positive and current variances non-negative.")
        rho = np.asarray(self.rho)
        d_b = np.sqrt(dt) * standard_normals[..., 0]
        d_w = np.sqrt(dt) * (rho * standard_normals[..., 0]
                            + np.sqrt(1.0 - rho**2) * standard_normals[..., 1])
        sqrt_v = np.sqrt(variance)
        next_log = (log_spot + (self.r - 0.5 * variance.sum(axis=-1)) * dt
                    + (sqrt_v * d_b).sum(axis=-1))
        next_variance = np.maximum(
            variance + np.asarray(self.kappa) * (np.asarray(self.theta) - variance) * dt
            + np.asarray(self.xi) * sqrt_v * d_w, 0.0)
        increments = np.stack([d_b, d_w], axis=-1).reshape(log_spot.shape + (4,))
        return next_log, next_variance, increments

    def simulate(self, n_paths: int, n_steps: int, rng: np.random.Generator):
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive.")
        dt = self.T / n_steps
        times = np.linspace(0.0, self.T, n_steps + 1)
        normals = rng.standard_normal((n_paths, n_steps, 2, 2))
        log_spot = np.empty((n_paths, n_steps + 1))
        variance = np.empty((n_paths, n_steps + 1, 2))
        increments = np.empty((n_paths, n_steps, 4))
        log_spot[:, 0], variance[:, 0] = np.log(self.s0), self.v0
        for k in range(n_steps):
            log_spot[:, k+1], variance[:, k+1], increments[:, k] = self.step(
                log_spot[:, k], variance[:, k], dt, normals[:, k])
        return TwoFactorHestonPaths(times, np.exp(log_spot), variance, increments)


@dataclass(frozen=True)
class TwoFactorHestonPaths:
    """Spot (n,N+1), factor variances (n,N+1,2), increments (n,N,4)."""

    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
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


def make_two_factor_heston_datasets(model, sizes, n_steps, seed=0):
    """Use the existing independent child-stream convention for each split."""
    children = np.random.SeedSequence(seed).spawn(len(sizes))
    return {name: model.simulate(size, n_steps, np.random.default_rng(child))
            for (name, size), child in zip(sizes.items(), children)}

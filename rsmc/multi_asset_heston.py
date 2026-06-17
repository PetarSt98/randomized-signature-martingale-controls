"""Three-asset Heston basket simulator under the risk-neutral measure Q.

Each asset has its own stochastic variance process and the six Brownian drivers
are simulated with a configurable correlation matrix. The discretisation mirrors
``rsmc.heston``: full-truncation Euler for variance and log-Euler for spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlations import correlated_standard_normals, validate_correlation_matrix


def _as_vector(name: str, values: tuple[float, ...] | np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def default_multi_asset_heston_correlation() -> np.ndarray:
    """Economically reasonable default correlation for three Heston assets.

    Driver order is ``S1, S2, S3, v1, v2, v3``. Spots are positively correlated,
    variance shocks are positively correlated, and each spot has negative
    leverage correlation with its own variance shock.
    """
    corr = np.eye(6)
    for i in range(3):
        for j in range(3):
            if i != j:
                corr[i, j] = 0.55
                corr[i + 3, j + 3] = 0.45
    for i in range(3):
        for j in range(3):
            corr[i, j + 3] = corr[j + 3, i] = -0.65 if i == j else -0.25
    return validate_correlation_matrix(corr, 6, name="default multi-asset Heston correlation")


@dataclass(frozen=True)
class MultiAssetHestonModel:
    """Three-asset Heston model with one variance factor per asset."""

    s0: tuple[float, float, float] = (100.0, 95.0, 105.0)
    v0: tuple[float, float, float] = (0.04, 0.05, 0.045)
    r: float = 0.03
    kappa: tuple[float, float, float] = (1.5, 1.3, 1.7)
    theta: tuple[float, float, float] = (0.04, 0.05, 0.045)
    xi: tuple[float, float, float] = (0.30, 0.35, 0.32)
    correlation: np.ndarray | None = None
    T: float = 1.0
    asset_names: tuple[str, str, str] = ("asset_1", "asset_2", "asset_3")

    def __post_init__(self) -> None:
        s0 = _as_vector("s0", self.s0, 3)
        v0 = _as_vector("v0", self.v0, 3)
        kappa = _as_vector("kappa", self.kappa, 3)
        theta = _as_vector("theta", self.theta, 3)
        xi = _as_vector("xi", self.xi, 3)
        if np.any(s0 <= 0):
            raise ValueError("s0 entries must be positive.")
        if np.any(v0 < 0):
            raise ValueError("v0 entries must be non-negative.")
        if np.any(kappa <= 0) or np.any(theta <= 0) or np.any(xi <= 0):
            raise ValueError("kappa, theta and xi entries must be positive.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if len(self.asset_names) != 3:
            raise ValueError("asset_names must contain exactly three names.")
        validate_correlation_matrix(
            default_multi_asset_heston_correlation() if self.correlation is None else self.correlation,
            6,
        )

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> "MultiAssetHestonPaths":
        """Simulate three correlated Heston assets on a uniform time grid."""
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}.")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}.")

        s0 = _as_vector("s0", self.s0, 3)
        v0 = _as_vector("v0", self.v0, 3)
        kappa = _as_vector("kappa", self.kappa, 3)
        theta = _as_vector("theta", self.theta, 3)
        xi = _as_vector("xi", self.xi, 3)
        corr = (
            default_multi_asset_heston_correlation()
            if self.correlation is None
            else validate_correlation_matrix(self.correlation, 6)
        )

        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, self.T, n_steps + 1)
        normals = correlated_standard_normals(rng, n_paths, n_steps, corr)
        brownian_increments = normals * sqrt_dt

        log_spot = np.empty((n_paths, n_steps + 1, 3))
        variance = np.empty((n_paths, n_steps + 1, 3))
        log_spot[:, 0, :] = np.log(s0)
        variance[:, 0, :] = v0

        for k in range(n_steps):
            v_k = variance[:, k, :]
            sqrt_v = np.sqrt(v_k)
            dw_spot = brownian_increments[:, k, :3]
            dw_var = brownian_increments[:, k, 3:]
            log_spot[:, k + 1, :] = (
                log_spot[:, k, :] + (self.r - 0.5 * v_k) * dt + sqrt_v * dw_spot
            )
            next_var = v_k + kappa * (theta - v_k) * dt + xi * sqrt_v * dw_var
            variance[:, k + 1, :] = np.maximum(next_var, 0.0)

        return MultiAssetHestonPaths(
            times=times,
            spot=np.exp(log_spot),
            variance=variance,
            brownian_increments=brownian_increments,
            asset_names=self.asset_names,
        )


@dataclass(frozen=True)
class MultiAssetHestonPaths:
    """Three-asset Heston paths and six Brownian martingale increments."""

    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
    brownian_increments: np.ndarray
    asset_names: tuple[str, str, str]

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])


def make_multi_asset_heston_datasets(
    model: MultiAssetHestonModel,
    sizes: dict[str, int],
    n_steps: int,
    seed: int = 0,
) -> dict[str, MultiAssetHestonPaths]:
    """Generate independent train / validation / test style datasets."""
    names = list(sizes)
    sub_seeds = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: model.simulate(sizes[name], n_steps, np.random.default_rng(sub_seed))
        for name, sub_seed in zip(names, sub_seeds)
    }

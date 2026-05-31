"""Commodity forward-curve simulator for an Asian crack-spread benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlations import correlated_standard_normals, validate_correlation_matrix


def _as_matrix(name: str, values: tuple[tuple[float, ...], ...] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def default_commodity_correlation() -> np.ndarray:
    """Default 9-driver correlation for three commodities and three curve nodes."""
    n_commodities, n_nodes = 3, 3
    loadings = np.zeros((n_commodities * n_nodes, 1 + n_commodities + n_nodes))
    for commodity in range(n_commodities):
        for node in range(n_nodes):
            row = commodity * n_nodes + node
            loadings[row, 0] = 0.45                       # broad energy factor
            loadings[row, 1 + commodity] = 0.45           # commodity-specific factor
            loadings[row, 1 + n_commodities + node] = 0.25

    corr = loadings @ loadings.T
    corr += np.diag(1.0 - np.diag(corr))
    return corr


@dataclass(frozen=True)
class CommodityForwardCurveModel:
    """Correlated lognormal futures curves for crude, gasoline and distillate."""

    initial_forwards: tuple[tuple[float, float, float], ...] = (
        (75.0, 74.0, 73.0),
        (94.0, 93.0, 92.0),
        (92.0, 91.0, 90.0),
    )
    volatilities: tuple[tuple[float, float, float], ...] = (
        (0.30, 0.26, 0.23),
        (0.36, 0.31, 0.28),
        (0.33, 0.29, 0.26),
    )
    r: float = 0.03
    correlation: np.ndarray | None = None
    T: float = 1.0
    commodity_names: tuple[str, str, str] = ("crude", "gasoline", "distillate")
    node_names: tuple[str, str, str] = ("node_1", "node_2", "node_3")

    def __post_init__(self) -> None:
        forwards = _as_matrix("initial_forwards", self.initial_forwards)
        volatilities = _as_matrix("volatilities", self.volatilities)
        if np.any(forwards <= 0):
            raise ValueError("initial_forwards entries must be positive.")
        if np.any(volatilities <= 0):
            raise ValueError("volatilities entries must be positive.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if len(self.commodity_names) != 3 or len(self.node_names) != 3:
            raise ValueError("commodity_names and node_names must each contain three names.")
        if self.correlation is not None:
            validate_correlation_matrix(self.correlation, 9)

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> "CommodityForwardCurvePaths":
        """Simulate exact lognormal martingale forward curves."""
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}.")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}.")

        forwards0 = _as_matrix("initial_forwards", self.initial_forwards)
        volatilities = _as_matrix("volatilities", self.volatilities)
        corr = (
            default_commodity_correlation()
            if self.correlation is None
            else validate_correlation_matrix(self.correlation, 9)
        )

        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, self.T, n_steps + 1)
        normals = correlated_standard_normals(rng, n_paths, n_steps, corr)
        brownian_increments = normals * sqrt_dt

        flat_initial = forwards0.reshape(-1)
        flat_vol = volatilities.reshape(-1)
        log_forwards = np.empty((n_paths, n_steps + 1, 9))
        log_forwards[:, 0, :] = np.log(flat_initial)
        drift = -0.5 * flat_vol**2 * dt

        for k in range(n_steps):
            log_forwards[:, k + 1, :] = (
                log_forwards[:, k, :] + drift + flat_vol * brownian_increments[:, k, :]
            )

        return CommodityForwardCurvePaths(
            times=times,
            forward_curves=np.exp(log_forwards).reshape(n_paths, n_steps + 1, 3, 3),
            brownian_increments=brownian_increments,
            commodity_names=self.commodity_names,
            node_names=self.node_names,
        )


@dataclass(frozen=True)
class CommodityForwardCurvePaths:
    """Forward curves and nine Brownian martingale increments."""

    times: np.ndarray
    forward_curves: np.ndarray
    brownian_increments: np.ndarray
    commodity_names: tuple[str, str, str]
    node_names: tuple[str, str, str]

    @property
    def n_paths(self) -> int:
        return self.forward_curves.shape[0]

    @property
    def n_steps(self) -> int:
        return self.forward_curves.shape[1] - 1

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])


def make_commodity_forward_curve_datasets(
    model: CommodityForwardCurveModel,
    sizes: dict[str, int],
    n_steps: int,
    seed: int = 0,
) -> dict[str, CommodityForwardCurvePaths]:
    """Generate independent train / validation / test style datasets."""
    names = list(sizes)
    sub_seeds = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: model.simulate(sizes[name], n_steps, np.random.default_rng(sub_seed))
        for name, sub_seed in zip(names, sub_seeds)
    }

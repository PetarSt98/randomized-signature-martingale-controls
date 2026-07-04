"""Experiment configuration and reproducible train / validation / test splits.

This freezes the experimental setup of the first benchmark from the proposal:

    * Model    : Black-Scholes (clean sanity-check model)
    * Payoff   : arithmetic Asian call
    * Time grid: N = 50 steps
    * Training paths   : 20_000
    * Validation paths :  5_000   (ridge lambda / reservoir scale tuning)
    * Test paths       : 100_000  (final unbiased evaluation)

The three samples must be *independent* so that sample-split estimation stays
unbiased (Theorem A). We get independence from a single master seed by spawning
separate child generators, which keeps the whole experiment reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .black_scholes import BlackScholesModel, SimulatedPaths


@dataclass(frozen=True)
class ExperimentConfig:
    """All knobs needed to generate the simulation datasets."""

    model: BlackScholesModel = field(default_factory=BlackScholesModel)
    strike: float = 100.0          # at-the-money call (K = S_0 by default)
    n_steps: int = 52             # N
    n_train: int = 10_000
    n_val: int = 5_000
    n_test: int = 50_000
    seed: int = 0                  # master seed; sub-samples derive from it


def make_datasets(config: ExperimentConfig) -> dict[str, SimulatedPaths]:
    """Generate independent train / validation / test path sets.

    Returns
    -------
    dict[str, SimulatedPaths]
        Keys ``"train"``, ``"val"`` and ``"test"``. Re-running with the same
        config reproduces the exact same paths.
    """
    train_seed, val_seed, test_seed = np.random.SeedSequence(config.seed).spawn(3)
    sizes = {
        "train": (config.n_train, train_seed),
        "val": (config.n_val, val_seed),
        "test": (config.n_test, test_seed),
    }
    return {
        name: config.model.simulate(
            n_paths=n_paths,
            n_steps=config.n_steps,
            rng=np.random.default_rng(seed),
        )
        for name, (n_paths, seed) in sizes.items()
    }

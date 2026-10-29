"""Asian tasks with declared observable summaries and fixed premiums."""
from __future__ import annotations

from dataclasses import asdict
import hashlib

import numpy as np

from .black_scholes import BlackScholesModel
from .heston import HestonModel, make_heston_datasets
from .multi_asset_heston import MultiAssetHestonModel, make_multi_asset_heston_datasets
from .two_factor_heston import TwoFactorHestonModel, make_two_factor_heston_datasets
from .hedging import hedging_features, multi_asset_hedging_features
from .payoffs import arithmetic_asian_call, basket_arithmetic_asian_call


def make_transfer_dataset(task, sizes, n_steps, seed, strike=100.):
    """Return data, times, rate, premium, metadata for an independent partition.

    The premium is the float32-representable training discounted payoff mean,
    frozen across all methods and cost conditions in this partition. Prices,
    variance factors and simulated innovations are never transferred between
    tasks. Existing tasks use price-history summaries; the optional two-factor
    task additionally observes total variance, never its individual components.
    """
    if task == "black_scholes":
        model = BlackScholesModel()
        children = np.random.SeedSequence(seed).spawn(len(sizes))
        paths = {name: model.simulate(size, n_steps, np.random.default_rng(child))
                 for (name, size), child in zip(sizes.items(), children)}
    elif task == "heston":
        model = HestonModel()
        paths = make_heston_datasets(model, sizes, n_steps, seed)
    elif task == "basket":
        model = MultiAssetHestonModel()
        paths = make_multi_asset_heston_datasets(model, sizes, n_steps, seed)
    elif task == "two_factor_heston":
        model = TwoFactorHestonModel()
        paths = make_two_factor_heston_datasets(model, sizes, n_steps, seed)
    else:
        raise ValueError(f"Unknown existing transfer task: {task}")
    basket = task == "basket"
    feature_builder = multi_asset_hedging_features if basket else hedging_features
    payoff_builder = basket_arithmetic_asian_call if basket else arithmetic_asian_call
    data = {name: dict(
        features=(hedging_features(p.spot, p.times, p.variance.sum(axis=-1) / 0.04,
                                  "total_variance") if task == "two_factor_heston"
                  else feature_builder(p.spot, p.times)),
        spot=p.spot, payoff=payoff_builder(p, strike, model.r))
        for name, p in paths.items()}
    payoff = data["train"]["payoff"]
    premium = float(np.float32(payoff.mean()))
    metadata = {
        "task": task, "dataset_seed": seed, "model": asdict(model), "strike": strike,
        "sizes": dict(sizes), "n_steps": n_steps,
        "feature_names": list(data["train"]["features"].names),
        "premium": premium,
        "premium_training_se": float(payoff.std(ddof=1) / np.sqrt(len(payoff))),
        "dataset_spot_sha256": {name: hashlib.sha256(p.spot.tobytes()).hexdigest()
                                for name, p in paths.items()},
        "premium_rule": "float32(training discounted payoff mean), frozen within partition",
        "gains": "positions times discounted traded-price increments, summed over dates and assets",
        "basket_weights": [1./3]*3 if basket else None,
        "payoff_includes_initial": False, "running_feature_includes_initial": True,
    }
    if task == "two_factor_heston":
        metadata["dataset_features_sha256"] = {
            name: hashlib.sha256(d["features"].values.tobytes()).hexdigest()
            for name, d in data.items()}
    return data, paths["train"].times, model.r, premium, metadata

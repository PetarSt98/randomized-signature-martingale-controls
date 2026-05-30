"""Randomized-Signature Martingale Control Variates (rsmc).

Building blocks implemented so far:

    * Black-Scholes Monte Carlo simulator (paths + Brownian innovations).
    * Arithmetic Asian call payoff (plus European call / closed-form check).
    * Asian-aware, time-augmented preprocessing into feature channels.
    * Frozen randomized-signature reservoir (the adapted state map h_k).

Still to come: the trained linear readout, the sample-split martingale-control
estimator, and the finite truncated-signature benchmark.
"""

from __future__ import annotations

from .black_scholes import BlackScholesModel, SimulatedPaths
from .config import ExperimentConfig, make_datasets
from .control import (
    ControlReport,
    RidgeMartingaleControl,
    assemble_control_features,
    control_features,
    evaluate_control,
    tune_ridge_lambda,
    variance_reduction,
)
from .features import PathFeatures, asian_features, heston_features
from .heston import HestonModel, HestonPaths, make_heston_datasets
from .payoffs import (
    arithmetic_asian_call,
    black_scholes_call_price,
    european_call,
    geometric_asian_call_price,
)
from .reservoir import RandomizedSignatureReservoir, readout_features
from .signature import (
    assemble_signature_control_features,
    running_signature,
    signature_dim,
)

__all__ = [
    "BlackScholesModel",
    "SimulatedPaths",
    "ExperimentConfig",
    "make_datasets",
    "arithmetic_asian_call",
    "european_call",
    "black_scholes_call_price",
    "geometric_asian_call_price",
    "PathFeatures",
    "asian_features",
    "heston_features",
    "HestonModel",
    "HestonPaths",
    "make_heston_datasets",
    "RandomizedSignatureReservoir",
    "readout_features",
    "control_features",
    "assemble_control_features",
    "RidgeMartingaleControl",
    "variance_reduction",
    "tune_ridge_lambda",
    "ControlReport",
    "evaluate_control",
    "signature_dim",
    "running_signature",
    "assemble_signature_control_features",
]

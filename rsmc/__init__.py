"""Randomized-Signature Martingale Control Variates (rsmc).

Building blocks include the Black-Scholes, Heston, three-asset Heston and
commodity forward-curve simulators; adapted feature maps; discounted payoffs;
the randomized-signature reservoir; ridge martingale controls; and finite
truncated-signature benchmark features.
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
from .commodity import (
    CommodityForwardCurveModel,
    CommodityForwardCurvePaths,
    make_commodity_forward_curve_datasets,
)
from .features import (
    PathFeatures,
    asian_features,
    commodity_crack_spread_features,
    heston_features,
    multi_asset_heston_features,
)
from .hedging import (
    HedgeReport,
    assemble_linear_hedge_features,
    assemble_quadratic_hedge_features,
    assemble_reservoir_hedge_features,
    assemble_signature_hedge_features,
    discounted_traded_increments,
    evaluate_hedge,
    fit_hedge,
    hedging_features,
    heston_variance_particle_filter,
    multi_asset_hedging_features,
    observe_variance,
    particle_filter_hedging_features,
)
from .heston import HestonModel, HestonPaths, make_heston_datasets
from .multi_asset_heston import (
    MultiAssetHestonModel,
    MultiAssetHestonPaths,
    make_multi_asset_heston_datasets,
)
from .payoffs import (
    arithmetic_asian_call,
    basket_arithmetic_asian_call,
    black_scholes_call_price,
    commodity_crack_spread_asian_call,
    european_call,
    geometric_asian_call_price,
)
from .reservoir import RandomizedSignatureReservoir, readout_features
from .signature import (
    assemble_signature_control_features,
    running_signature,
    signature_dim,
)
from .transformer_hedging import (
    TransformerHedgeConfig,
    evaluate_transformer_hedge,
    fit_transformer_hedge,
    transformer_available,
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
    "multi_asset_heston_features",
    "commodity_crack_spread_features",
    "HestonModel",
    "HestonPaths",
    "make_heston_datasets",
    "MultiAssetHestonModel",
    "MultiAssetHestonPaths",
    "make_multi_asset_heston_datasets",
    "CommodityForwardCurveModel",
    "CommodityForwardCurvePaths",
    "make_commodity_forward_curve_datasets",
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
    "basket_arithmetic_asian_call",
    "commodity_crack_spread_asian_call",
    "observe_variance",
    "discounted_traded_increments",
    "hedging_features",
    "multi_asset_hedging_features",
    "heston_variance_particle_filter",
    "particle_filter_hedging_features",
    "assemble_linear_hedge_features",
    "assemble_quadratic_hedge_features",
    "assemble_reservoir_hedge_features",
    "assemble_signature_hedge_features",
    "HedgeReport",
    "evaluate_hedge",
    "fit_hedge",
    "TransformerHedgeConfig",
    "transformer_available",
    "fit_transformer_hedge",
    "evaluate_transformer_hedge",
]

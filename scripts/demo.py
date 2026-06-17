# %% [markdown]
# # Randomized-Signature Martingale Control Variate - benchmark demo
#
# One command runs the final benchmark ladder and compares frozen randomized
# reservoirs against finite truncated signatures:
#
# * Black-Scholes arithmetic Asian call, d = 3, m = 1
# * Heston arithmetic Asian call, d = 4, m = 2
# * three-asset Heston basket Asian call, d = 8, m = 6
# * commodity forward-curve Asian crack-spread option, d = 12, m = 9
#
# For each case we sweep reservoirs D = 16, 32, 64, 128, 256 and exact
# signatures L = 2, 3, 4 through the same train / validation / test split, ridge
# fit, martingale-control assembly and reporting. Very wide exact-signature rows
# can be disabled with the flags below; skipped rows are printed with their
# theoretical dimensions.

# %%
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


def find_repo_root() -> Path:
    """Locate the project root (the folder containing ``rsmc``)."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:                       # running as a notebook cell
        start = Path.cwd().resolve()
    for folder in [start, *start.parents]:
        if (folder / "rsmc").is_dir():
            return folder
    return start


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmc import (  # noqa: E402
    CommodityForwardCurveModel,
    ExperimentConfig,
    HestonModel,
    MultiAssetHestonModel,
    PathFeatures,
    RandomizedSignatureReservoir,
    RidgeMartingaleControl,
    arithmetic_asian_call,
    asian_features,
    assemble_control_features,
    assemble_signature_control_features,
    basket_arithmetic_asian_call,
    commodity_crack_spread_asian_call,
    commodity_crack_spread_features,
    evaluate_control,
    heston_features,
    make_commodity_forward_curve_datasets,
    make_datasets,
    make_heston_datasets,
    make_multi_asset_heston_datasets,
    multi_asset_heston_features,
    signature_dim,
    tune_ridge_lambda,
)


RIDGE_LAMBDAS = (
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    300.0,
    500.0,
    1_000.0,
    5_000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
)
RESERVOIR_DIMS = (16, 32, 64, 128, 256) 
SIGNATURE_LEVELS = (2, 3, 4)
RUN_SIGNATURE_D8_L4 = False
RUN_SIGNATURE_D12_L3 = False
RUN_SIGNATURE_D12_L4 = False
DEFAULT_N_STEPS = 252
DEFAULT_SIZES = {"train": 10_000, "val": 3_000, "test": 10_000}
SMOKE_SIZES = {"train": 200, "val": 100, "test": 300}


def martingale_dim(increments: np.ndarray) -> int:
    """Infer the martingale increment dimension from the simulated shocks."""
    return 1 if increments.ndim == 2 else increments.shape[2]


def path_shape_summary(paths) -> str:
    """Compact shape summary for a simulator output object."""
    parts = []
    if hasattr(paths, "spot"):
        parts.append(f"spot {paths.spot.shape}")
    if hasattr(paths, "variance"):
        parts.append(f"variance {paths.variance.shape}")
    if hasattr(paths, "forward_curves"):
        parts.append(f"forward_curves {paths.forward_curves.shape}")
    parts.append(f"dM {paths.brownian_increments.shape}")
    return ", ".join(parts)


def build_benchmarks(
    sizes: dict[str, int],
    n_steps: int,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Build the four benchmark cases with independent Monte Carlo splits."""
    bs_config = ExperimentConfig(
        n_train=sizes["train"],
        n_val=sizes["val"],
        n_test=sizes["test"],
        n_steps=n_steps,
        seed=seed,
    )
    bs_model = bs_config.model

    heston_model = HestonModel()
    multi_heston_model = MultiAssetHestonModel()
    commodity_model = CommodityForwardCurveModel()

    return [
        {
            "model": "Black-Scholes",
            "product": "Asian call",
            "datasets": make_datasets(bs_config),
            "feature_builder": asian_features,
            "payoff_builder": lambda paths: arithmetic_asian_call(
                paths, strike=bs_config.strike, rate=bs_model.r
            ),
            "reservoir_seed": seed + 100,
        },
        {
            "model": "Heston",
            "product": "Asian call",
            "datasets": make_heston_datasets(heston_model, sizes, n_steps, seed=seed + 10),
            "feature_builder": heston_features,
            "payoff_builder": lambda paths: arithmetic_asian_call(
                paths, strike=100.0, rate=heston_model.r
            ),
            "reservoir_seed": seed + 200,
        },
        {
            "model": "three-asset Heston",
            "product": "basket Asian call",
            "datasets": make_multi_asset_heston_datasets(
                multi_heston_model, sizes, n_steps, seed=seed + 20
            ),
            "feature_builder": multi_asset_heston_features,
            "payoff_builder": lambda paths: basket_arithmetic_asian_call(
                paths, strike=100.0, rate=multi_heston_model.r
            ),
            "reservoir_seed": seed + 300,
        },
        {
            "model": "commodity forward curve",
            "product": "Asian crack spread",
            "datasets": make_commodity_forward_curve_datasets(
                commodity_model, sizes, n_steps, seed=seed + 30
            ),
            "feature_builder": commodity_crack_spread_features,
            "payoff_builder": lambda paths: commodity_crack_spread_asian_call(
                paths, strike=15.0, rate=commodity_model.r
            ),
            "reservoir_seed": seed + 400,
        },
    ]


def prepare_case(case: dict[str, object]) -> dict[str, object]:
    """Build features and discounted payoffs once for one benchmark case."""
    datasets = case["datasets"]
    feature_builder: Callable = case["feature_builder"]  # type: ignore[assignment]
    payoff_builder: Callable = case["payoff_builder"]    # type: ignore[assignment]
    features: dict[str, PathFeatures] = {
        name: feature_builder(paths) for name, paths in datasets.items()
    }
    payoff = {name: payoff_builder(paths) for name, paths in datasets.items()}
    test_payoff = payoff["test"]
    return {
        **case,
        "features": features,
        "payoff": payoff,
        "d": features["train"].n_channels,
        "m": martingale_dim(datasets["train"].brownian_increments),
        "plain_price": float(test_payoff.mean()),
        "plain_se": float(test_payoff.std(ddof=1) / np.sqrt(test_payoff.size)),
    }


def make_bundles(
    prepared: dict[str, object],
    sizes: dict[str, int],
) -> dict[str, tuple[PathFeatures, np.ndarray, np.ndarray, np.ndarray]]:
    """Build sliced train / val / test bundles for one method row."""
    datasets = prepared["datasets"]
    features = prepared["features"]
    payoff = prepared["payoff"]
    bundles = {}
    for name in ("train", "val", "test"):
        n = sizes[name]
        paths = datasets[name]
        feats = features[name]
        bundles[name] = (
            PathFeatures(feats.values[:n], feats.names),
            paths.times,
            paths.brownian_increments[:n],
            payoff[name][:n],
        )
    return bundles


def run_control_row(
    prepared: dict[str, object],
    method: str,
    label: str,
    feature_dim: int,
    assemble: Callable[[PathFeatures, np.ndarray, np.ndarray], np.ndarray],
    sizes: dict[str, int],
    lambdas: tuple[float, ...],
) -> dict[str, object]:
    """Assemble Phi, tune ridge, fit and evaluate one method row."""
    bundles = make_bundles(prepared, sizes)

    phi_train = assemble(*bundles["train"][:3])
    phi_val = assemble(*bundles["val"][:3])
    start = time.perf_counter()
    phi_test = assemble(*bundles["test"][:3])
    feature_test_s = time.perf_counter() - start

    start = time.perf_counter()
    best_lambda, _ = tune_ridge_lambda(
        phi_train, bundles["train"][3],
        phi_val, bundles["val"][3],
        lambdas,
    )
    control = RidgeMartingaleControl.fit(phi_train, bundles["train"][3], best_lambda)
    fit_s = time.perf_counter() - start

    report = evaluate_control(control, phi_test, bundles["test"][3])
    return {
        "model": prepared["model"],
        "product": prepared["product"],
        "d": prepared["d"],
        "m": prepared["m"],
        "method": method,
        "label": label,
        "feature_dim": feature_dim,
        "q": phi_test.shape[1],
        "n_train": sizes["train"],
        "n_val": sizes["val"],
        "n_eval": sizes["test"],
        "plain_price": report.price_mc,
        "plain_se": report.se_mc,
        "controlled_price": report.price_cv,
        "controlled_se": report.se_cv,
        "var_mc": report.var_mc,
        "var_cv": report.var_cv,
        "variance_reduction": report.variance_reduction,
        "se_reduction": report.se_reduction,
        "control_mean": report.control_mean,
        "best_lambda": best_lambda,
        "feature_test_s": feature_test_s,
        "fit_s": fit_s,
    }


def reservoir_assembler(prepared: dict[str, object], dim: int) -> Callable:
    """Create a frozen reservoir assembler for one benchmark case and D."""
    reservoir = RandomizedSignatureReservoir.draw(
        n_channels=int(prepared["d"]),
        dim=dim,
        rng=np.random.default_rng(int(prepared["reservoir_seed"]) + dim),
    )
    return lambda feats, times, shocks: assemble_control_features(
        reservoir, feats, times, shocks
    )


def signature_assembler(level: int) -> Callable:
    """Create a finite-signature assembler for one level."""
    return lambda feats, times, shocks: assemble_signature_control_features(
        feats, times, shocks, level=level
    )


def should_run_signature_level(d: int, level: int) -> bool:
    """Whether the demo should run this exact-signature row."""
    if d == 8 and level == 4:
        return RUN_SIGNATURE_D8_L4
    if d == 12 and level == 3:
        return RUN_SIGNATURE_D12_L3
    if d == 12 and level == 4:
        return RUN_SIGNATURE_D12_L4
    return True


def run_case_sweep(
    prepared: dict[str, object],
    sizes: dict[str, int],
    reservoir_dims: tuple[int, ...],
    signature_levels: tuple[int, ...],
    lambdas: tuple[float, ...],
    print_output: bool = True,
) -> list[dict[str, object]]:
    """Run all reservoir and signature rows for one benchmark case."""
    d = int(prepared["d"])
    m = int(prepared["m"])
    rows: list[dict[str, object]] = []

    if print_output:
        case_payoff = prepared["payoff"]["test"][:sizes["test"]]
        print(f"\n{prepared['model']} - {prepared['product']}")
        print(f"  feature channels ({d}): {', '.join(prepared['features']['train'].names)}")
        print(f"  martingale dimension m = {m}")
        print(f"  train / val / test for this case: {sizes}")
        print(f"  test path shapes: {path_shape_summary(prepared['datasets']['test'])}")
        print(
            f"  plain MC on case test: price {case_payoff.mean():.4f}, "
            f"SE {case_payoff.std(ddof=1) / np.sqrt(case_payoff.size):.4f}"
        )

    for dim in reservoir_dims:
        if print_output:
            print(f"  reservoir D={dim} ...", flush=True)
        rows.append(
            run_control_row(
                prepared=prepared,
                method="reservoir",
                label=f"reservoir D={dim}",
                feature_dim=dim,
                assemble=reservoir_assembler(prepared, dim),
                sizes=sizes,
                lambdas=lambdas,
            )
        )

    for level in signature_levels:
        if not should_run_signature_level(d, level):
            if print_output:
                sig_dim = signature_dim(d, level)
                q = m * (2 + sig_dim)
                print(
                    f"  signature L={level} skipped "
                    f"(dim={sig_dim:,}, q={q:,}; enable RUN_SIGNATURE_D{d}_L{level} to run)",
                    flush=True,
                )
            continue
        if print_output:
            print(f"  signature L={level} ...", flush=True)
        rows.append(
            run_control_row(
                prepared=prepared,
                method="signature",
                label=f"signature L={level}",
                feature_dim=signature_dim(d, level),
                assemble=signature_assembler(level),
                sizes=sizes,
                lambdas=lambdas,
            )
        )

    if print_output:
        print_method_table(rows)
    return rows


def print_method_table(rows: list[dict[str, object]]) -> None:
    """Print one per-case reservoir-vs-signature table."""
    best_resid = min(float(row["var_cv"]) for row in rows)
    header = (
        f"{'method':<17}{'dim':>7}{'q':>9}{'n':>7}{'price':>10}{'SE':>9}"
        f"{'VR':>9}{'SE x':>8}{'resid/best':>12}{'mean(C)':>11}"
        f"{'lambda':>10}{'feat(s)':>9}{'fit(s)':>8}"
    )
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for row in rows:
        print(
            "  "
            f"{str(row['label']):<17}"
            f"{int(row['feature_dim']):>7}"
            f"{int(row['q']):>9}"
            f"{int(row['n_eval']):>7}"
            f"{float(row['controlled_price']):>10.4f}"
            f"{float(row['controlled_se']):>9.4f}"
            f"{float(row['variance_reduction']):>8.2f}x"
            f"{float(row['se_reduction']):>7.2f}x"
            f"{float(row['var_cv']) / best_resid:>11.2f}x"
            f"{float(row['control_mean']):>11.4f}"
            f"{float(row['best_lambda']):>10.1g}"
            f"{float(row['feature_test_s']):>9.2f}"
            f"{float(row['fit_s']):>8.2f}"
        )
    print("  " + "-" * len(header))


def print_signature_dimension_note(d_values: list[int]) -> None:
    """Show why high-d exact signatures are expensive."""
    print("\nSignature dimensions q_L = d + ... + d^L (excluding level 0):")
    for d in d_values:
        dims = ", ".join(f"L={level}: {signature_dim(d, level):,}" for level in SIGNATURE_LEVELS)
        print(f"  d={d:>2}: {dims}")
    print("The d=12 commodity case is the main randomized-signature compression benchmark.")


def print_final_summary(rows: list[dict[str, object]]) -> None:
    """Print one summary row per benchmark case."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["product"])), []).append(row)

    header = (
        f"{'model':<25}{'product':<22}{'d':>4}{'m':>4}"
        f"{'best reservoir':>20}{'VR':>9}{'feat':>8}"
        f"{'best signature':>20}{'VR':>9}{'feat':>8}"
    )
    print("\nFINAL BENCHMARK SUMMARY")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for (model, product), case_rows in groups.items():
        reservoirs = [row for row in case_rows if row["method"] == "reservoir"]
        signatures = [row for row in case_rows if row["method"] == "signature"]
        best_reservoir = max(reservoirs, key=lambda row: float(row["variance_reduction"]))
        best_signature = (
            max(signatures, key=lambda row: float(row["variance_reduction"]))
            if signatures else None
        )
        signature_label = "not run" if best_signature is None else str(best_signature["label"])
        signature_vr = np.nan if best_signature is None else float(best_signature["variance_reduction"])
        signature_feat = np.nan if best_signature is None else float(best_signature["feature_test_s"])
        print(
            f"{model:<25}{product:<22}"
            f"{int(best_reservoir['d']):>4}{int(best_reservoir['m']):>4}"
            f"{str(best_reservoir['label']):>20}"
            f"{float(best_reservoir['variance_reduction']):>8.2f}x"
            f"{float(best_reservoir['feature_test_s']):>8.2f}"
            f"{signature_label:>20}"
            f"{signature_vr:>8.2f}x"
            f"{signature_feat:>8.2f}"
        )
    print("-" * len(header))
    print("All rows use the same train/val/test size, so reservoir and signature comparisons are matched.")


def run_demo(
    sizes: dict[str, int] | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    reservoir_dims: tuple[int, ...] = RESERVOIR_DIMS,
    signature_levels: tuple[int, ...] = SIGNATURE_LEVELS,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    print_output: bool = True,
) -> list[dict[str, object]]:
    """Run the full reservoir-vs-signature sweep for all four cases."""
    sizes = DEFAULT_SIZES if sizes is None else sizes
    if print_output:
        print(f"train / val / test sizes: {sizes}; time steps N = {n_steps}")
        print(f"reservoir D sweep: {reservoir_dims}")
        print(f"signature L sweep: {signature_levels}")
        print(
            "signature row flags: "
            f"RUN_SIGNATURE_D8_L4={RUN_SIGNATURE_D8_L4}, "
            f"RUN_SIGNATURE_D12_L3={RUN_SIGNATURE_D12_L3}, "
            f"RUN_SIGNATURE_D12_L4={RUN_SIGNATURE_D12_L4}"
        )

    all_rows: list[dict[str, object]] = []
    prepared_cases = [prepare_case(case) for case in build_benchmarks(sizes, n_steps)]
    for prepared in prepared_cases:
        all_rows.extend(
            run_case_sweep(
                prepared=prepared,
                sizes=sizes,
                reservoir_dims=reservoir_dims,
                signature_levels=signature_levels,
                lambdas=lambdas,
                print_output=print_output,
            )
        )

    if print_output:
        print_final_summary(all_rows)
        print_signature_dimension_note([int(case["d"]) for case in prepared_cases])
    return all_rows


if __name__ == "__main__":
    run_demo()

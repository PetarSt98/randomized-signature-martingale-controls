# %% [markdown]
# # Observed-filtration hedging demo
#
# This mirrors ``scripts/control_variates_demo.py`` but changes the mathematical object:
#
# * pricing control variates integrate against simulator martingale increments dM;
# * hedging integrates against the discounted traded stock increment dS~.
#
# The Heston variance is hidden. The trader observes the stock and a noisy
# variance signal ``Y = V + eta eps``. The full-information row using true ``V``
# is reported only as an oracle reference; the reservoir and signature rows use
# the observed path ``(S, Y)`` and are therefore observed-filtration strategies.

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
    HestonModel,
    MultiAssetHestonModel,
    PathFeatures,
    RandomizedSignatureReservoir,
    arithmetic_asian_call,
    assemble_control_features,
    assemble_signature_control_features,
    basket_arithmetic_asian_call,
    discounted_traded_increments,
    evaluate_hedge,
    fit_hedge,
    hedging_features,
    make_heston_datasets,
    make_multi_asset_heston_datasets,
    multi_asset_hedging_features,
    observe_variance,
    signature_dim,
)
from rsmc.hedging import assemble_linear_hedge_features  # noqa: E402


MODEL = HestonModel()
MULTI_ASSET_MODEL = MultiAssetHestonModel()
STRIKE = 100.0
ETA_BASE = 0.02
ETA_SWEEP = (0.0, 0.01, 0.02, 0.04, 0.08)
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
RUN_SIGNATURE_D8_L3 = True
RUN_SIGNATURE_D8_L4 = False
RUN_SIGNATURE_D12_L3 = False
RUN_SIGNATURE_D12_L4 = False
RUN_ORACLE_RESERVOIRS = True
RUN_MULTI_ASSET_HEDGING = True
DEFAULT_N_STEPS = 52
DEFAULT_SIZES = {"train": 10_000, "val": 3_000, "test": 10_000}
SMOKE_SIZES = {"train": 200, "val": 100, "test": 300}
RESERVOIR_SEED = 2_024


def build_data(
    sizes: dict[str, int],
    n_steps: int,
    eta: float = ETA_BASE,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate independent Heston splits and attach observed hedge data."""
    datasets = make_heston_datasets(MODEL, sizes, n_steps, seed=seed)
    noise_seeds = np.random.SeedSequence(seed + 1).spawn(len(datasets))
    splits = {}
    for (name, paths), noise_seed in zip(datasets.items(), noise_seeds):
        signal = observe_variance(paths.variance, eta, np.random.default_rng(noise_seed))
        splits[name] = {
            "times": paths.times,
            "spot": paths.spot,
            "variance": paths.variance,
            "signal": signal,
            "payoff": arithmetic_asian_call(paths, STRIKE, MODEL.r),
            "dS": discounted_traded_increments(paths.spot, paths.times, MODEL.r),
        }
    return splits


def build_multi_asset_data(
    sizes: dict[str, int],
    n_steps: int,
    eta: float = ETA_BASE,
    seed: int = 100,
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate independent three-asset Heston splits and attach noisy signals."""
    datasets = make_multi_asset_heston_datasets(MULTI_ASSET_MODEL, sizes, n_steps, seed=seed)
    noise_seeds = np.random.SeedSequence(seed + 1).spawn(len(datasets))
    splits = {}
    for (name, paths), noise_seed in zip(datasets.items(), noise_seeds):
        signal = observe_variance(paths.variance, eta, np.random.default_rng(noise_seed))
        splits[name] = {
            "times": paths.times,
            "spot": paths.spot,
            "variance": paths.variance,
            "signal": signal,
            "payoff": basket_arithmetic_asian_call(paths, STRIKE, MULTI_ASSET_MODEL.r),
            "dS": discounted_traded_increments(paths.spot, paths.times, MULTI_ASSET_MODEL.r),
        }
    return splits


def price_only_features(split: dict[str, np.ndarray]) -> PathFeatures:
    """Observable price-history features, no variance signal."""
    return hedging_features(split["spot"], split["times"])


def observed_signal_features(split: dict[str, np.ndarray]) -> PathFeatures:
    """Observed-filtration features: price history plus noisy variance signal."""
    return hedging_features(
        split["spot"],
        split["times"],
        split["signal"] - MODEL.v0,
        "signal",
    )


def oracle_features(split: dict[str, np.ndarray]) -> PathFeatures:
    """Full-information reference features using the true hidden variance."""
    return hedging_features(
        split["spot"],
        split["times"],
        split["variance"] - MODEL.v0,
        "variance",
    )


def multi_price_only_features(split: dict[str, np.ndarray]) -> PathFeatures:
    """Observable three-stock price-history features, no variance signal."""
    return multi_asset_hedging_features(split["spot"], split["times"])


def multi_observed_signal_features(split: dict[str, np.ndarray]) -> PathFeatures:
    """Observed three-stock features plus three noisy variance signals."""
    return multi_asset_hedging_features(
        split["spot"],
        split["times"],
        split["signal"] - np.asarray(MULTI_ASSET_MODEL.v0)[None, None, :],
        "signal",
    )


def multi_oracle_features(split: dict[str, np.ndarray]) -> PathFeatures:
    """Full-information reference features using three true hidden variances."""
    return multi_asset_hedging_features(
        split["spot"],
        split["times"],
        split["variance"] - np.asarray(MULTI_ASSET_MODEL.v0)[None, None, :],
        "variance",
    )


def prepare_case(
    sizes: dict[str, int],
    n_steps: int,
    eta: float,
    seed: int = 0,
) -> dict[str, object]:
    """Build all feature sets, payoffs and traded increments for the hedge case."""
    splits = build_data(sizes, n_steps, eta, seed)
    feature_builders = {
        "price": price_only_features,
        "observed": observed_signal_features,
        "oracle": oracle_features,
    }
    features = {
        key: {name: builder(split) for name, split in splits.items()}
        for key, builder in feature_builders.items()
    }
    payoff = {name: split["payoff"] for name, split in splits.items()}
    increments = {name: split["dS"] for name, split in splits.items()}
    times = {name: split["times"] for name, split in splits.items()}
    test_payoff = payoff["test"]
    return {
        "model": "Heston hidden variance",
        "product": "Asian call hedge",
        "eta": eta,
        "features": features,
        "payoff": payoff,
        "increments": increments,
        "times": times,
        "d": features["observed"]["train"].n_channels,
        "m": 1,                          # one traded stock increment dS~
        "plain_price": float(test_payoff.mean()),
        "plain_std": float(test_payoff.std(ddof=1)),
        "reservoir_seed": seed + RESERVOIR_SEED,
    }


def prepare_multi_asset_case(
    sizes: dict[str, int],
    n_steps: int,
    eta: float,
    seed: int = 100,
) -> dict[str, object]:
    """Build the three-asset Heston observed-filtration basket hedge case."""
    splits = build_multi_asset_data(sizes, n_steps, eta, seed)
    feature_builders = {
        "price": multi_price_only_features,
        "observed": multi_observed_signal_features,
        "oracle": multi_oracle_features,
    }
    features = {
        key: {name: builder(split) for name, split in splits.items()}
        for key, builder in feature_builders.items()
    }
    payoff = {name: split["payoff"] for name, split in splits.items()}
    increments = {name: split["dS"] for name, split in splits.items()}
    times = {name: split["times"] for name, split in splits.items()}
    test_payoff = payoff["test"]
    return {
        "model": "three-asset Heston hidden variance",
        "product": "basket Asian hedge",
        "eta": eta,
        "features": features,
        "payoff": payoff,
        "increments": increments,
        "times": times,
        "d": features["observed"]["train"].n_channels,
        "m": 3,                          # three traded discounted stock increments
        "plain_price": float(test_payoff.mean()),
        "plain_std": float(test_payoff.std(ddof=1)),
        "reservoir_seed": seed + RESERVOIR_SEED,
    }


def make_bundles(
    prepared: dict[str, object],
    sizes: dict[str, int],
    feature_key: str,
) -> dict[str, tuple[PathFeatures, np.ndarray, np.ndarray, np.ndarray]]:
    """Build sliced train / val / test bundles for one hedge row."""
    features = prepared["features"][feature_key]       # type: ignore[index]
    times = prepared["times"]                          # type: ignore[assignment]
    increments = prepared["increments"]                # type: ignore[assignment]
    payoff = prepared["payoff"]                        # type: ignore[assignment]
    bundles = {}
    for name in ("train", "val", "test"):
        n = sizes[name]
        feats = features[name]
        bundles[name] = (
            PathFeatures(feats.values[:n], feats.names),
            times[name],
            increments[name][:n],
            payoff[name][:n],
        )
    return bundles


def run_hedge_row(
    prepared: dict[str, object],
    method: str,
    label: str,
    feature_key: str,
    feature_dim: int,
    assemble: Callable[[PathFeatures, np.ndarray, np.ndarray], np.ndarray],
    sizes: dict[str, int],
    lambdas: tuple[float, ...],
) -> dict[str, object]:
    """Assemble Psi, tune ridge, fit and evaluate one hedge row."""
    bundles = make_bundles(prepared, sizes, feature_key)

    psi_train = assemble(*bundles["train"][:3])
    psi_val = assemble(*bundles["val"][:3])
    start = time.perf_counter()
    psi_test = assemble(*bundles["test"][:3])
    feature_test_s = time.perf_counter() - start

    start = time.perf_counter()
    control, best_lambda = fit_hedge(
        psi_train,
        bundles["train"][3],
        psi_val,
        bundles["val"][3],
        lambdas,
    )
    fit_s = time.perf_counter() - start

    validation_error = bundles["val"][3] - control.intercept - control.control(psi_val)
    validation_mse = float(np.mean(validation_error**2))
    report = evaluate_hedge(control, psi_test, bundles["test"][3])
    return {
        "model": prepared["model"],
        "product": prepared["product"],
        "eta": prepared["eta"],
        "d": prepared["features"][feature_key]["train"].n_channels,  # type: ignore[index]
        "m": prepared["m"],
        "method": method,
        "label": label,
        "feature_key": feature_key,
        "feature_dim": feature_dim,
        "q": psi_test.shape[1],
        "n_train": sizes["train"],
        "n_val": sizes["val"],
        "n_eval": sizes["test"],
        "price": report.price,
        "capital": report.capital,
        "payoff_std": report.payoff_std,
        "error_std": report.error_std,
        "rmse": report.rmse,
        "error_mse": report.error_mse,
        "validation_mse": validation_mse,
        "error_var": report.error_var,
        "variance_reduction": report.variance_reduction,
        "mean_error": report.mean_error,
        "mean_error_se": report.mean_error_se,
        "best_lambda": best_lambda,
        "feature_test_s": feature_test_s,
        "fit_s": fit_s,
    }


def reservoir_assembler(prepared: dict[str, object], dim: int) -> Callable:
    """Create a frozen observed-path reservoir assembler for one D."""
    reservoir = RandomizedSignatureReservoir.draw(
        n_channels=int(prepared["d"]),
        dim=dim,
        rng=np.random.default_rng(int(prepared["reservoir_seed"]) + dim),
    )
    return lambda feats, times, increments: assemble_control_features(
        reservoir, feats, times, increments
    )


def signature_assembler(level: int) -> Callable:
    """Create a finite observed-path signature assembler for one level."""
    return lambda feats, times, increments: assemble_signature_control_features(
        feats, times, increments, level=level
    )


def should_run_signature_level(d: int, level: int) -> bool:
    """Whether the demo should run this exact-signature row."""
    if d == 8 and level == 3:
        return RUN_SIGNATURE_D8_L3
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
    """Run the baseline, reservoir and signature hedge rows."""
    d = int(prepared["d"])
    rows: list[dict[str, object]] = []

    if print_output:
        observed_names = prepared["features"]["observed"]["train"].names  # type: ignore[index]
        print(f"\n{prepared['model']} - {prepared['product']}")
        print(f"  observed feature channels ({d}): {', '.join(observed_names)}")
        print(f"  traded martingale dimension m = {int(prepared['m'])} (discounted stock increments dS~)")
        print(f"  observation noise eta = {float(prepared['eta']):.4f}")
        print(f"  train / val / test for this case: {sizes}")
        print(
            f"  unhedged payoff on case test: price {float(prepared['plain_price']):.4f}, "
            f"std {float(prepared['plain_std']):.4f}"
        )

    baseline_specs = (
        ("price-only linear", "price"),
        ("observed signal linear", "observed"),
        ("oracle true V linear", "oracle"),
    )
    for label, feature_key in baseline_specs:
        if print_output:
            print(f"  {label} ...", flush=True)
        feature_dim = prepared["features"][feature_key]["train"].n_channels  # type: ignore[index]
        rows.append(
            run_hedge_row(
                prepared=prepared,
                method="baseline",
                label=label,
                feature_key=feature_key,
                feature_dim=feature_dim,
                assemble=assemble_linear_hedge_features,
                sizes=sizes,
                lambdas=lambdas,
            )
        )

    if RUN_ORACLE_RESERVOIRS:
        for dim in reservoir_dims:
            if print_output:
                print(f"  oracle true V reservoir D={dim} ...", flush=True)
            rows.append(
                run_hedge_row(
                    prepared=prepared,
                    method="oracle_reservoir",
                    label=f"oracle reservoir D={dim}",
                    feature_key="oracle",
                    feature_dim=dim,
                    assemble=reservoir_assembler(prepared, dim),
                    sizes=sizes,
                    lambdas=lambdas,
                )
            )

    for dim in reservoir_dims:
        if print_output:
            print(f"  reservoir D={dim} ...", flush=True)
        rows.append(
            run_hedge_row(
                prepared=prepared,
                method="reservoir",
                label=f"reservoir D={dim}",
                feature_key="observed",
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
                q = int(prepared["m"]) * (2 + sig_dim)
                print(
                    f"  signature L={level} skipped "
                    f"(dim={sig_dim:,}, q={q:,}; enable RUN_SIGNATURE_D{d}_L{level} to run)",
                    flush=True,
                )
            continue
        if print_output:
            print(f"  signature L={level} ...", flush=True)
        rows.append(
            run_hedge_row(
                prepared=prepared,
                method="signature",
                label=f"signature L={level}",
                feature_key="observed",
                feature_dim=signature_dim(d, level),
                assemble=signature_assembler(level),
                sizes=sizes,
                lambdas=lambdas,
            )
        )

    if print_output:
        print_hedge_table(rows)
    return rows


def print_hedge_table(rows: list[dict[str, object]]) -> None:
    """Print the observed-filtration hedge comparison table."""
    best_error_mse = min(float(row["error_mse"]) for row in rows)
    header = (
        f"{'method':<23}{'dim':>7}{'q':>8}{'n':>7}{'capital':>10}{'price':>9}"
        f"{'std(H)':>9}{'std(e)':>9}{'RMSE':>9}{'valRMSE':>9}{'VR':>9}"
        f"{'risk/best':>11}{'mean(e)':>10}{'lambda':>10}{'feat(s)':>9}{'fit(s)':>8}"
    )
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for row in rows:
        print(
            "  "
            f"{str(row['label']):<23}"
            f"{int(row['feature_dim']):>7}"
            f"{int(row['q']):>8}"
            f"{int(row['n_eval']):>7}"
            f"{float(row['capital']):>10.4f}"
            f"{float(row['price']):>9.4f}"
            f"{float(row['payoff_std']):>9.4f}"
            f"{float(row['error_std']):>9.4f}"
            f"{float(row['rmse']):>9.4f}"
            f"{np.sqrt(float(row['validation_mse'])):>9.4f}"
            f"{float(row['variance_reduction']):>8.2f}x"
            f"{float(row['error_mse']) / best_error_mse:>10.2f}x"
            f"{float(row['mean_error']):>10.4f}"
            f"{float(row['best_lambda']):>10.1g}"
            f"{float(row['feature_test_s']):>9.2f}"
            f"{float(row['fit_s']):>8.2f}"
        )
    print("  " + "-" * len(header))


def print_signature_dimension_note(d_values: list[int]) -> None:
    """Show the finite-signature sizes for the observed feature dimensions."""
    print("\nSignature dimensions q_L = d + ... + d^L (excluding level 0):")
    for d in d_values:
        dims = ", ".join(f"L={level}: {signature_dim(d, level):,}" for level in SIGNATURE_LEVELS)
        print(f"  d={d:>2}: {dims}")


def selected_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Validation-selected headline rows and test gap-closure diagnostics."""
    reservoirs = [row for row in rows if row["method"] == "reservoir"]
    signatures = [row for row in rows if row["method"] == "signature"]
    oracle_candidates = [
        row for row in rows
        if row["label"] == "oracle true V linear" or row["method"] == "oracle_reservoir"
    ]
    oracle = min(oracle_candidates, key=lambda row: float(row["validation_mse"]))
    observed = next(row for row in rows if row["label"] == "observed signal linear")
    best_reservoir = min(reservoirs, key=lambda row: float(row["validation_mse"]))
    best_signature = (
        min(signatures, key=lambda row: float(row["validation_mse"]))
        if signatures else None
    )

    raw_mse = float(observed["error_mse"])
    oracle_mse = float(oracle["error_mse"])
    denom = raw_mse - oracle_mse

    def gap(row: dict[str, object] | None) -> float:
        if row is None or denom <= 0.0:
            return np.nan
        return (raw_mse - float(row["error_mse"])) / denom

    return {
        "model": rows[0]["model"],
        "product": rows[0]["product"],
        "d": best_reservoir["d"],
        "eta": best_reservoir["eta"],
        "observed": observed,
        "oracle": oracle,
        "best_reservoir": best_reservoir,
        "best_signature": best_signature,
        "reservoir_gap": gap(best_reservoir),
        "signature_gap": gap(best_signature),
    }


def print_final_summary(rows: list[dict[str, object]]) -> None:
    """Print one summary row for the hedging benchmark."""
    def format_gap(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.1%}"

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["product"])), []).append(row)

    header = (
        f"{'model':<38}{'product':<22}{'d':>4}{'eta':>8}"
        f"{'oracle':>24}{'RMSE':>9}{'observed RMSE':>15}"
        f"{'best reservoir':>20}{'RMSE':>9}{'gap':>8}{'feat':>8}"
        f"{'best signature':>20}{'RMSE':>9}{'gap':>8}{'feat':>8}"
    )
    print("\nFINAL HEDGING SUMMARY")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for (model, product), case_rows in groups.items():
        summary = selected_summary(case_rows)
        oracle = summary["oracle"]
        observed = summary["observed"]
        best_reservoir = summary["best_reservoir"]
        best_signature = summary["best_signature"]
        signature_label = "not run" if best_signature is None else str(best_signature["label"])
        signature_rmse = np.nan if best_signature is None else float(best_signature["rmse"])
        signature_feat = np.nan if best_signature is None else float(best_signature["feature_test_s"])
        reservoir_gap = float(summary["reservoir_gap"])
        signature_gap = float(summary["signature_gap"])
        print(
            f"{model:<38}{product:<22}"
            f"{int(best_reservoir['d']):>4}"
            f"{float(best_reservoir['eta']):>8.3f}"
            f"{str(oracle['label']):>24}"
            f"{float(oracle['rmse']):>9.4f}"
            f"{float(observed['rmse']):>15.4f}"
            f"{str(best_reservoir['label']):>20}"
            f"{float(best_reservoir['rmse']):>9.4f}"
            f"{format_gap(reservoir_gap):>8}"
            f"{float(best_reservoir['feature_test_s']):>8.2f}"
            f"{signature_label:>20}"
            f"{signature_rmse:>9.4f}"
            f"{format_gap(signature_gap):>8}"
            f"{signature_feat:>8.2f}"
        )
    print("-" * len(header))
    print("Best reservoir/signature/oracle rows are selected by validation MSE; test RMSE is then reported.")
    print("Gap is normalized test-MSE closure versus observed-signal linear and the selected true-V oracle.")
    print("Reservoir and signature rows use only observed (S, Y) history; the oracle row uses true V only as a reference.")


def run_demo(
    sizes: dict[str, int] | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    eta: float = ETA_BASE,
    reservoir_dims: tuple[int, ...] = RESERVOIR_DIMS,
    signature_levels: tuple[int, ...] = SIGNATURE_LEVELS,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    print_output: bool = True,
) -> list[dict[str, object]]:
    """Run the observed-filtration reservoir-vs-signature hedging sweep."""
    sizes = DEFAULT_SIZES if sizes is None else sizes
    if print_output:
        print(f"train / val / test sizes: {sizes}; time steps N = {n_steps}")
        print(f"reservoir D sweep: {reservoir_dims}")
        print(f"signature L sweep: {signature_levels}")
        print(
            "signature row flags: "
            f"RUN_SIGNATURE_D8_L3={RUN_SIGNATURE_D8_L3}, "
            f"RUN_SIGNATURE_D8_L4={RUN_SIGNATURE_D8_L4}, "
            f"RUN_SIGNATURE_D12_L3={RUN_SIGNATURE_D12_L3}, "
            f"RUN_SIGNATURE_D12_L4={RUN_SIGNATURE_D12_L4}"
        )
        print(f"multi-asset d=8 extension: RUN_MULTI_ASSET_HEDGING={RUN_MULTI_ASSET_HEDGING}")

    prepared_cases = [prepare_case(sizes, n_steps, eta)]
    if RUN_MULTI_ASSET_HEDGING:
        prepared_cases.append(prepare_multi_asset_case(sizes, n_steps, eta))

    rows: list[dict[str, object]] = []
    for prepared in prepared_cases:
        rows.extend(
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
        print_final_summary(rows)
        print_signature_dimension_note([int(case["d"]) for case in prepared_cases])
    return rows


def run_eta_sweep(
    etas: tuple[float, ...] = ETA_SWEEP,
    sizes: dict[str, int] | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    reservoir_dims: tuple[int, ...] = RESERVOIR_DIMS,
    signature_levels: tuple[int, ...] = SIGNATURE_LEVELS,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    print_output: bool = True,
) -> list[dict[str, object]]:
    """Run the central hedging diagnostic: test RMSE versus observation noise."""
    sizes = DEFAULT_SIZES if sizes is None else sizes
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for eta in etas:
        rows = run_demo(
            sizes=sizes,
            n_steps=n_steps,
            eta=eta,
            reservoir_dims=reservoir_dims,
            signature_levels=signature_levels,
            lambdas=lambdas,
            print_output=False,
        )
        all_rows.extend(rows)
        groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            groups.setdefault((str(row["model"]), str(row["product"])), []).append(row)
        summaries.extend(selected_summary(case_rows) for case_rows in groups.values())

    if print_output:
        def format_gap(value: float) -> str:
            if not np.isfinite(value):
                return "n/a"
            return f"{value:.1%}"

        header = (
            f"{'model':<38}{'d':>4}{'eta':>8}{'observed RMSE':>15}{'oracle RMSE':>13}"
            f"{'reservoir':>20}{'RMSE':>9}{'gap':>8}"
            f"{'signature':>20}{'RMSE':>9}{'gap':>8}"
        )
        print("\nOBSERVATION-NOISE SWEEP")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for summary in summaries:
            observed = summary["observed"]
            oracle = summary["oracle"]
            reservoir = summary["best_reservoir"]
            signature = summary["best_signature"]
            signature_label = "not run" if signature is None else str(signature["label"])
            signature_rmse = np.nan if signature is None else float(signature["rmse"])
            print(
                f"{str(summary['model']):<38}"
                f"{int(summary['d']):>4}"
                f"{float(summary['eta']):>8.3f}"
                f"{float(observed['rmse']):>15.4f}"
                f"{float(oracle['rmse']):>13.4f}"
                f"{str(reservoir['label']):>20}"
                f"{float(reservoir['rmse']):>9.4f}"
                f"{format_gap(float(summary['reservoir_gap'])):>8}"
                f"{signature_label:>20}"
                f"{signature_rmse:>9.4f}"
                f"{format_gap(float(summary['signature_gap'])):>8}"
            )
        print("-" * len(header))
        print("Rows are selected by validation MSE at each eta; RMSE/gap are measured on the independent test split.")
    return all_rows


if __name__ == "__main__":
    run_demo()

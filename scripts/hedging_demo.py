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
    basket_arithmetic_asian_call,
    discounted_traded_increments,
    evaluate_hedge,
    fit_hedge,
    hedging_features,
    make_heston_datasets,
    make_multi_asset_heston_datasets,
    multi_asset_hedging_features,
    observe_variance,
    heston_variance_particle_filter,
    particle_filter_hedging_features,
    assemble_reservoir_hedge_features,
    assemble_signature_hedge_features,
    assemble_quadratic_hedge_features,
    signature_dim,
    TransformerHedgeConfig,
    evaluate_transformer_hedge,
    fit_transformer_hedge,
    transformer_available,
)
from rsmc.hedging import assemble_linear_hedge_features  # noqa: E402


MODEL = HestonModel()
MULTI_ASSET_MODEL = MultiAssetHestonModel()
STRIKE = 100.0
ETA_BASE = 0.02
ETA_SWEEP = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16)
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
RUN_PARTICLE_FILTER = True
PARTICLE_FILTER_PARTICLES = 256
RUN_TRANSFORMER_HEDGE = True
RUN_ORACLE_TRANSFORMER = True
RUN_PAIRED_INTERVALS = True
PAIRED_BOOTSTRAP_REPLICATIONS = 1_000
PAIRED_BOOTSTRAP_SEED = 91_337
TRANSFORMER_D_MODEL = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_WINDOW = 16
TRANSFORMER_EPOCHS = 20
TRANSFORMER_BATCH_SIZE = 256
TRANSFORMER_LR = 1e-3
TRANSFORMER_AVAILABLE = transformer_available()
DEFAULT_N_STEPS = 252
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
        "particle_filter_seed": seed + 80_000,
        "splits": splits,
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


def ensure_particle_filter_features(prepared: dict[str, object]) -> None:
    """Build the one-asset Heston particle-filter features on demand."""
    features = prepared["features"]  # type: ignore[assignment]
    if "particle_filter" in features:
        return
    splits = prepared.get("splits")
    if splits is None:
        raise ValueError("particle filter features are only available for the one-asset Heston case.")

    seed_sequence = np.random.SeedSequence(
        int(prepared["particle_filter_seed"]) + PARTICLE_FILTER_PARTICLES
    )
    split_seeds = seed_sequence.spawn(len(splits))  # type: ignore[arg-type]
    particle_features = {}
    timings = {}
    for (name, split), seed in zip(splits.items(), split_seeds):  # type: ignore[union-attr]
        start = time.perf_counter()
        mean, var = heston_variance_particle_filter(
            MODEL,
            split["spot"],
            split["signal"],
            split["times"],
            eta=float(prepared["eta"]),
            n_particles=PARTICLE_FILTER_PARTICLES,
            rng=np.random.default_rng(seed),
        )
        particle_features[name] = particle_filter_hedging_features(
            split["spot"],
            split["times"],
            split["signal"],
            mean,
            var,
            variance_center=MODEL.v0,
        )
        timings[name] = time.perf_counter() - start
    features["particle_filter"] = particle_features
    prepared["particle_filter_feature_s_by_split"] = timings


def particle_filter_timing(prepared: dict[str, object], split: str) -> float:
    timings = prepared.get("particle_filter_feature_s_by_split", {})
    return float(timings.get(split, 0.0))  # type: ignore[union-attr]


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
    if report.errors is None:
        raise RuntimeError("hedge evaluator did not return pathwise errors.")
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
        "errors": report.errors,
        "best_lambda": best_lambda,
        "feature_test_s": feature_test_s,
        "fit_s": fit_s,
    }


def run_transformer_row(
    prepared: dict[str, object],
    feature_key: str,
    sizes: dict[str, int],
    config: TransformerHedgeConfig,
    method: str = "transformer",
    label_prefix: str = "transformer",
) -> dict[str, object]:
    """Train and evaluate one causal sliding-window Transformer hedge row."""
    bundles = make_bundles(prepared, sizes, feature_key)

    fitted = fit_transformer_hedge(
        bundles["train"][0],
        bundles["train"][2],
        bundles["train"][3],
        bundles["val"][0],
        bundles["val"][2],
        bundles["val"][3],
        config,
    )

    start = time.perf_counter()
    report = evaluate_transformer_hedge(
        fitted,
        bundles["test"][0],
        bundles["test"][2],
        bundles["test"][3],
    )
    if report.errors is None:
        raise RuntimeError("transformer evaluator did not return pathwise errors.")
    feature_test_s = time.perf_counter() - start
    return {
        "model": prepared["model"],
        "product": prepared["product"],
        "eta": prepared["eta"],
        "d": prepared["features"][feature_key]["train"].n_channels,  # type: ignore[index]
        "m": prepared["m"],
        "method": method,
        "label": f"{label_prefix} w={config.window}",
        "feature_key": feature_key,
        "feature_dim": config.d_model,
        "q": fitted.n_parameters,
        "n_train": sizes["train"],
        "n_val": sizes["val"],
        "n_eval": sizes["test"],
        "price": report.price,
        "capital": report.capital,
        "payoff_std": report.payoff_std,
        "error_std": report.error_std,
        "rmse": report.rmse,
        "error_mse": report.error_mse,
        "validation_mse": fitted.validation_mse,
        "error_var": report.error_var,
        "variance_reduction": report.variance_reduction,
        "mean_error": report.mean_error,
        "mean_error_se": report.mean_error_se,
        "errors": report.errors,
        "best_lambda": np.nan,
        "feature_test_s": feature_test_s,
        "fit_s": fitted.fit_s,
    }


def reservoir_assembler(prepared: dict[str, object], dim: int) -> Callable:
    """Create a frozen observed-path reservoir assembler for one D."""
    reservoir = RandomizedSignatureReservoir.draw(
        n_channels=int(prepared["d"]),
        dim=dim,
        rng=np.random.default_rng(int(prepared["reservoir_seed"]) + dim),
    )
    return lambda feats, times, increments: assemble_reservoir_hedge_features(
        reservoir, feats, times, increments
    )


def signature_assembler(level: int) -> Callable:
    """Create a finite observed-path signature assembler for one level."""
    return lambda feats, times, increments: assemble_signature_hedge_features(
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
    """Run the baseline, reservoir, signature and transformer hedge rows."""
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

    if "splits" in prepared:
        if print_output:
            print("  oracle true V quadratic ...", flush=True)
        feature_dim = prepared["features"]["oracle"]["train"].n_channels  # type: ignore[index]
        rows.append(
            run_hedge_row(
                prepared=prepared,
                method="oracle_quadratic",
                label="oracle true V quadratic",
                feature_key="oracle",
                feature_dim=feature_dim,
                assemble=assemble_quadratic_hedge_features,
                sizes=sizes,
                lambdas=lambdas,
            )
        )

    if RUN_PARTICLE_FILTER and "splits" in prepared:
        if print_output:
            print(f"  particle filter P={PARTICLE_FILTER_PARTICLES} ...", flush=True)
        ensure_particle_filter_features(prepared)
        feature_dim = prepared["features"]["particle_filter"]["train"].n_channels  # type: ignore[index]
        for label, assemble in (
            (f"particle filter affine P={PARTICLE_FILTER_PARTICLES}", assemble_linear_hedge_features),
            (f"particle filter quadratic P={PARTICLE_FILTER_PARTICLES}", assemble_quadratic_hedge_features),
        ):
            row = run_hedge_row(
                prepared=prepared,
                method="particle_filter",
                label=label,
                feature_key="particle_filter",
                feature_dim=feature_dim,
                assemble=assemble,
                sizes=sizes,
                lambdas=lambdas,
            )
            row["feature_test_s"] = float(row["feature_test_s"]) + particle_filter_timing(
                prepared, "test"
            )
            row["filter_train_s"] = particle_filter_timing(prepared, "train")
            row["filter_val_s"] = particle_filter_timing(prepared, "val")
            row["filter_test_s"] = particle_filter_timing(prepared, "test")
            rows.append(row)

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
                q = int(prepared["m"]) * (d + 1 + sig_dim)
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

    if RUN_TRANSFORMER_HEDGE:
        if TRANSFORMER_AVAILABLE:
            config = TransformerHedgeConfig(
                d_model=TRANSFORMER_D_MODEL,
                n_heads=TRANSFORMER_HEADS,
                n_layers=TRANSFORMER_LAYERS,
                window=TRANSFORMER_WINDOW,
                learning_rate=TRANSFORMER_LR,
                epochs=TRANSFORMER_EPOCHS,
                batch_size=TRANSFORMER_BATCH_SIZE,
                seed=int(prepared["reservoir_seed"]) + 70_000,
            )
            if print_output:
                print(
                    f"  transformer w={config.window}, d_model={config.d_model}, "
                    f"layers={config.n_layers}, heads={config.n_heads} ...",
                    flush=True,
                )
            rows.append(
                run_transformer_row(
                    prepared=prepared,
                    feature_key="observed",
                    sizes=sizes,
                    config=config,
                )
            )
            if RUN_ORACLE_TRANSFORMER:
                if print_output:
                    print(
                        f"  oracle transformer w={config.window}, d_model={config.d_model}, "
                        f"layers={config.n_layers}, heads={config.n_heads} ...",
                        flush=True,
                    )
                rows.append(
                    run_transformer_row(
                        prepared=prepared,
                        feature_key="oracle",
                        sizes=sizes,
                        config=config,
                        method="oracle_transformer",
                        label_prefix="oracle transformer",
                    )
                )
        elif print_output:
            print("  transformer skipped (PyTorch not installed; install torch to enable)", flush=True)

    if print_output:
        print_hedge_table(rows)
    return rows


def print_hedge_table(rows: list[dict[str, object]]) -> None:
    """Print the observed-filtration hedge comparison table."""
    best_error_mse = min(float(row["error_mse"]) for row in rows)
    header = (
        f"{'method':<32}{'dim':>7}{'q/params':>10}{'n':>7}{'capital':>10}{'price':>9}"
        f"{'std(H)':>9}{'std(e)':>9}{'RMSE':>9}{'valRMSE':>9}{'VR':>9}"
        f"{'risk/best':>11}{'mean(e)':>10}{'lambda':>10}{'feat(s)':>9}{'fit(s)':>8}"
    )
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for row in rows:
        best_lambda = float(row["best_lambda"])
        lambda_text = "n/a" if not np.isfinite(best_lambda) else f"{best_lambda:.1g}"
        print(
            "  "
            f"{str(row['label']):<32}"
            f"{int(row['feature_dim']):>7}"
            f"{int(row['q']):>10}"
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
            f"{lambda_text:>10}"
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
    transformers = [row for row in rows if row["method"] == "transformer"]
    particle_filters = [row for row in rows if row["method"] == "particle_filter"]
    oracle_candidates = [
        row for row in rows
        if row["label"] == "oracle true V linear"
        or row["method"] == "oracle_quadratic"
        or row["method"] == "oracle_reservoir"
        or row["method"] == "oracle_transformer"
    ]
    oracle = min(oracle_candidates, key=lambda row: float(row["validation_mse"]))
    observed = next(row for row in rows if row["label"] == "observed signal linear")
    best_reservoir = min(reservoirs, key=lambda row: float(row["validation_mse"]))
    best_signature = (
        min(signatures, key=lambda row: float(row["validation_mse"]))
        if signatures else None
    )
    best_transformer = (
        min(transformers, key=lambda row: float(row["validation_mse"]))
        if transformers else None
    )
    best_particle_filter = (
        min(particle_filters, key=lambda row: float(row["validation_mse"]))
        if particle_filters else None
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
        "best_transformer": best_transformer,
        "best_particle_filter": best_particle_filter,
        "reservoir_gap": gap(best_reservoir),
        "signature_gap": gap(best_signature),
        "transformer_gap": gap(best_transformer),
        "particle_filter_gap": gap(best_particle_filter),
    }


def _row_errors(row: dict[str, object]) -> np.ndarray:
    errors = row.get("errors")
    if errors is None:
        raise ValueError(f"row {row.get('label')} does not contain pathwise errors.")
    return np.asarray(errors, dtype=float)


def _format_interval(point: float, lo: float, hi: float) -> tuple[str, str]:
    if not np.isfinite(point) or not np.isfinite(lo) or not np.isfinite(hi):
        return "n/a", "[n/a, n/a]"
    return f"{point:.4f}", f"[{lo:.4f}, {hi:.4f}]"


def _bootstrap_delta_mse(
    row_a: dict[str, object],
    row_b: dict[str, object],
    n_bootstrap: int,
    rng: np.random.Generator,
    batch_size: int = 200,
) -> tuple[float, float, float]:
    """Bootstrap ``MSE(row_a) - MSE(row_b)`` on paired test paths."""
    diff = _row_errors(row_a) ** 2 - _row_errors(row_b) ** 2
    n = diff.shape[0]
    point = float(np.mean(diff))
    samples = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        samples[start:stop] = np.mean(diff[indices], axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


def _bootstrap_gap_close(
    candidate: dict[str, object],
    raw: dict[str, object],
    full_info: dict[str, object],
    n_bootstrap: int,
    rng: np.random.Generator,
    batch_size: int = 200,
) -> tuple[float, float, float, float, float, float]:
    """Bootstrap ``(MSE_raw - MSE_candidate) / (MSE_raw - MSE_full)``."""
    candidate_sq = _row_errors(candidate) ** 2
    raw_sq = _row_errors(raw) ** 2
    full_sq = _row_errors(full_info) ** 2
    n = raw_sq.shape[0]

    raw_mse = float(np.mean(raw_sq))
    candidate_mse = float(np.mean(candidate_sq))
    full_mse = float(np.mean(full_sq))
    denom = raw_mse - full_mse
    point = np.nan if denom <= 0.0 else (raw_mse - candidate_mse) / denom

    samples = np.empty(n_bootstrap)
    denom_samples = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        raw_boot = np.mean(raw_sq[indices], axis=1)
        candidate_boot = np.mean(candidate_sq[indices], axis=1)
        full_boot = np.mean(full_sq[indices], axis=1)
        denom_boot = raw_boot - full_boot
        denom_samples[start:stop] = denom_boot
        samples[start:stop] = np.where(
            denom_boot > 0.0,
            (raw_boot - candidate_boot) / denom_boot,
            np.nan,
        )
    valid_samples = np.isfinite(samples)
    if np.any(valid_samples):
        lo, hi = np.percentile(samples[valid_samples], [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    denom_lo, denom_hi = np.percentile(denom_samples, [2.5, 97.5])
    valid_fraction = float(np.mean(denom_samples > 0.0))
    return float(point), float(lo), float(hi), valid_fraction, float(denom_lo), float(denom_hi)


def _matched_oracle_reservoir(
    rows: list[dict[str, object]],
    reservoir: dict[str, object],
) -> dict[str, object] | None:
    target_label = str(reservoir["label"]).replace("reservoir", "oracle reservoir", 1)
    return next((row for row in rows if row["label"] == target_label), None)


def print_paired_interval_summary(groups: dict[tuple[str, str], list[dict[str, object]]]) -> None:
    """Print paired bootstrap MSE-difference intervals for selected rows."""
    header = (
        f"{'model':<38}{'d':>4}{'eta':>8}  {'comparison':<46}"
        f"{'Delta MSE':>12}{'95% CI':>24}"
    )
    rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED)

    print("\nPAIRED TEST-PATH MSE INTERVALS")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for case_rows in groups.values():
        summary = selected_summary(case_rows)
        reservoir = summary["best_reservoir"]
        comparisons = [
            ("reservoir - raw signal linear", reservoir, summary["observed"]),
            ("reservoir - best signature", reservoir, summary["best_signature"]),
            ("reservoir - transformer", reservoir, summary["best_transformer"]),
            ("reservoir - particle filter", reservoir, summary["best_particle_filter"]),
            (
                "reservoir - matched full-info reservoir",
                reservoir,
                _matched_oracle_reservoir(case_rows, reservoir),
            ),
        ]
        for label, left, right in comparisons:
            if right is None:
                continue
            point, lo, hi = _bootstrap_delta_mse(
                left,
                right,
                PAIRED_BOOTSTRAP_REPLICATIONS,
                rng,
            )
            point_text, interval_text = _format_interval(point, lo, hi)
            print(
                f"{str(summary['model']):<38}"
                f"{int(summary['d']):>4}"
                f"{float(summary['eta']):>8.3f}  "
                f"{label:<46}"
                f"{point_text:>12}"
                f"{interval_text:>24}"
            )
    print("-" * len(header))
    print("Negative Delta MSE means the selected reservoir has lower test MSE than the benchmark.")

    gap_header = (
        f"{'model':<38}{'d':>4}{'eta':>8}{'reservoir':>20}"
        f"{'matched full info':>28}{'GapClose':>12}{'95% CI':>24}{'valid denom':>14}"
    )
    print("\nMATCHED RESERVOIR GAPCLOSE INTERVALS")
    print("-" * len(gap_header))
    print(gap_header)
    print("-" * len(gap_header))
    for case_rows in groups.values():
        summary = selected_summary(case_rows)
        reservoir = summary["best_reservoir"]
        matched = _matched_oracle_reservoir(case_rows, reservoir)
        if matched is None:
            continue
        point, lo, hi, valid_fraction, denom_lo, _ = _bootstrap_gap_close(
            reservoir,
            summary["observed"],
            matched,
            PAIRED_BOOTSTRAP_REPLICATIONS,
            rng,
        )
        if valid_fraction >= 0.95 and denom_lo > 0.0:
            point_text, interval_text = _format_interval(point, lo, hi)
        else:
            point_text, interval_text = "n/a", "unstable / not reported"
        print(
            f"{str(summary['model']):<38}"
            f"{int(summary['d']):>4}"
            f"{float(summary['eta']):>8.3f}"
            f"{str(reservoir['label']):>20}"
            f"{str(matched['label']):>28}"
            f"{point_text:>12}"
            f"{interval_text:>24}"
            f"{valid_fraction:>14.1%}"
        )
    print("-" * len(gap_header))
    print("Bootstrap intervals are conditional on the fitted models and use paired test-path resampling.")


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
        f"{'best reservoir':>20}{'RMSE':>9}{'gap':>10}{'feat':>8}"
        f"{'best signature':>20}{'RMSE':>9}{'gap':>10}{'feat':>8}"
        f"  {'particle filter':>28}{'RMSE':>9}{'gap':>10}{'feat':>8}"
        f"  {'transformer':>20}{'RMSE':>9}{'gap':>10}{'fit':>8}"
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
        best_transformer = summary["best_transformer"]
        best_particle_filter = summary["best_particle_filter"]
        signature_label = "not run" if best_signature is None else str(best_signature["label"])
        signature_rmse = np.nan if best_signature is None else float(best_signature["rmse"])
        signature_feat = np.nan if best_signature is None else float(best_signature["feature_test_s"])
        pf_label = "not run" if best_particle_filter is None else str(best_particle_filter["label"])
        pf_rmse = np.nan if best_particle_filter is None else float(best_particle_filter["rmse"])
        pf_feat = np.nan if best_particle_filter is None else float(best_particle_filter["feature_test_s"])
        transformer_label = "not run" if best_transformer is None else str(best_transformer["label"])
        transformer_rmse = np.nan if best_transformer is None else float(best_transformer["rmse"])
        transformer_fit = np.nan if best_transformer is None else float(best_transformer["fit_s"])
        reservoir_gap = float(summary["reservoir_gap"])
        signature_gap = float(summary["signature_gap"])
        particle_filter_gap = float(summary["particle_filter_gap"])
        transformer_gap = float(summary["transformer_gap"])
        print(
            f"{model:<38}{product:<22}"
            f"{int(best_reservoir['d']):>4}"
            f"{float(best_reservoir['eta']):>8.3f}"
            f"{str(oracle['label']):>24}"
            f"{float(oracle['rmse']):>9.4f}"
            f"{float(observed['rmse']):>15.4f}"
            f"{str(best_reservoir['label']):>20}"
            f"{float(best_reservoir['rmse']):>9.4f}"
            f"{format_gap(reservoir_gap):>10}"
            f"{float(best_reservoir['feature_test_s']):>8.2f}"
            f"{signature_label:>20}"
            f"{signature_rmse:>9.4f}"
            f"{format_gap(signature_gap):>10}"
            f"{signature_feat:>8.2f}"
            f"  {pf_label:>28}"
            f"{pf_rmse:>9.4f}"
            f"{format_gap(particle_filter_gap):>10}"
            f"{pf_feat:>8.2f}"
            f"  {transformer_label:>20}"
            f"{transformer_rmse:>9.4f}"
            f"{format_gap(transformer_gap):>10}"
            f"{transformer_fit:>8.2f}"
        )
    print("-" * len(header))
    print("Best reservoir/signature/oracle rows are selected by validation MSE; test RMSE is then reported.")
    print("Gap is normalized test-MSE closure versus observed-signal linear and the selected true-V oracle.")
    print("Reservoir, signature, particle-filter and transformer rows use only observed history; true V is oracle-only.")
    if RUN_PAIRED_INTERVALS:
        print_paired_interval_summary(groups)


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
        print(
            "particle filter: "
            f"RUN_PARTICLE_FILTER={RUN_PARTICLE_FILTER}, "
            f"particles={PARTICLE_FILTER_PARTICLES} (one-asset Heston only)"
        )
        print(
            "paired intervals: "
            f"RUN_PAIRED_INTERVALS={RUN_PAIRED_INTERVALS}, "
            f"bootstraps={PAIRED_BOOTSTRAP_REPLICATIONS}"
        )
        print(
            "transformer hedge: "
            f"RUN_TRANSFORMER_HEDGE={RUN_TRANSFORMER_HEDGE}, "
            f"RUN_ORACLE_TRANSFORMER={RUN_ORACLE_TRANSFORMER}, "
            f"available={TRANSFORMER_AVAILABLE}, "
            f"d_model={TRANSFORMER_D_MODEL}, layers={TRANSFORMER_LAYERS}, "
            f"heads={TRANSFORMER_HEADS}, window={TRANSFORMER_WINDOW}, "
            f"epochs={TRANSFORMER_EPOCHS}"
        )

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
            f"{'reservoir':>20}{'RMSE':>9}{'gap':>10}"
            f"{'signature':>20}{'RMSE':>9}{'gap':>10}"
            f"  {'particle filter':>28}{'RMSE':>9}{'gap':>10}"
            f"  {'transformer':>20}{'RMSE':>9}{'gap':>10}"
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
            particle_filter = summary["best_particle_filter"]
            transformer = summary["best_transformer"]
            signature_label = "not run" if signature is None else str(signature["label"])
            signature_rmse = np.nan if signature is None else float(signature["rmse"])
            pf_label = "not run" if particle_filter is None else str(particle_filter["label"])
            pf_rmse = np.nan if particle_filter is None else float(particle_filter["rmse"])
            transformer_label = "not run" if transformer is None else str(transformer["label"])
            transformer_rmse = np.nan if transformer is None else float(transformer["rmse"])
            print(
                f"{str(summary['model']):<38}"
                f"{int(summary['d']):>4}"
                f"{float(summary['eta']):>8.3f}"
                f"{float(observed['rmse']):>15.4f}"
                f"{float(oracle['rmse']):>13.4f}"
                f"{str(reservoir['label']):>20}"
                f"{float(reservoir['rmse']):>9.4f}"
                f"{format_gap(float(summary['reservoir_gap'])):>10}"
                f"{signature_label:>20}"
                f"{signature_rmse:>9.4f}"
                f"{format_gap(float(summary['signature_gap'])):>10}"
                f"  {pf_label:>28}"
                f"{pf_rmse:>9.4f}"
                f"{format_gap(float(summary['particle_filter_gap'])):>10}"
                f"  {transformer_label:>20}"
                f"{transformer_rmse:>9.4f}"
                f"{format_gap(float(summary['transformer_gap'])):>10}"
            )
        print("-" * len(header))
        print("Rows are selected by validation MSE at each eta; RMSE/gap are measured on the independent test split.")
        if RUN_PAIRED_INTERVALS:
            interval_groups: dict[tuple[str, str, float], list[dict[str, object]]] = {}
            for row in all_rows:
                interval_groups.setdefault(
                    (str(row["model"]), str(row["product"]), float(row["eta"])),
                    [],
                ).append(row)
            print_paired_interval_summary(interval_groups)
    return all_rows


if __name__ == "__main__":
    run_demo()

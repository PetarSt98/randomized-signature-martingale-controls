# %% [markdown]
# # Randomized-Signature Martingale Control Variate - demo + signature benchmark
#
# Petar Stojkovic - CAS Machine Learning in Finance and Insurance, ETH Zurich
#
# One file, run top to bottom like a notebook. It has two parts:
#
# **Part A - the method (cells 1-7).** Price an arithmetic Asian call by Monte
# Carlo, then subtract a zero-mean *martingale control* learned from a *frozen
# randomized-signature reservoir* (an RNN with frozen random weights + a trained
# linear readout). Result: ~50x variance reduction, still unbiased.
#
# **Part B - the benchmark (cells 8-11).** Replace the reservoir with the *real*
# finite truncated signature and run the **same** readout / fit / evaluation, to
# answer: does the cheap random reservoir match the real signature, and what does
# it cost? We compare across reservoir size `D` and signature level `L`, and at a
# richer `d = 5` path representation where the signature dimension `q_L = d^L`
# explodes.
#
# Same control-variate architecture, different adapted path representation:
#
# * reservoir : `h_k`            -> linear readout -> `Z_k = beta . [1, t_k, h_k]`
# * signature : `Sig^{<=L}(x_0:k)` -> linear readout -> `Z_k = alpha . [1, t_k, s_k]`
#
# then `C = sum_k Z_k * dW_{k+1}` in both cases.
#
# How to run: open in Cursor / VS Code and run the cells, or
# `python scripts/demo.py` from the repo root. Part B is the heavy part
# (a few minutes); the `# %%` cells let you run Part A alone.

# %%
# --- imports and configuration -------------------------------------------
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

from rsmc import (
    ExperimentConfig,
    HestonModel,
    RandomizedSignatureReservoir,
    RidgeMartingaleControl,
    arithmetic_asian_call,
    asian_features,
    assemble_control_features,
    assemble_signature_control_features,
    black_scholes_call_price,
    european_call,
    evaluate_control,
    geometric_asian_call_price,
    heston_features,
    make_datasets,
    make_heston_datasets,
    signature_dim,
    tune_ridge_lambda,
    variance_reduction,
)
from rsmc.features import PathFeatures

FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

config = ExperimentConfig()                  # DATA knobs: model, strike, N, sample sizes, seed

# METHOD / BENCHMARK knobs. Deliberately *not* in ExperimentConfig: that dataclass
# configures only data generation (which paths to simulate), while these configure
# the estimator and the sweep. They live here, together and inline, so the file
# reads like a notebook - one place to find every dial, no values duplicated below.
RESERVOIR_DIM = 64                           # Part A: reservoir size D
RIDGE_LAMBDAS = (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1)
SWEEP_DIMS = (16, 32, 64, 128, 256)                   # Part B: reservoir sizes D to compare
SWEEP_LEVELS = (2, 3, 4)                     # Part B: signature truncation levels L
N_BENCH = 30_000                             # Part B: shared test paths for the d=3 and d=5 sweeps
print(config)

# %% [markdown]
# ## 1 - Simulate paths
# Independent train / validation / test sets, all derived from one master seed
# so the whole experiment is reproducible.

# %%
datasets = make_datasets(config)
payoff = {
    name: arithmetic_asian_call(paths, strike=config.strike, rate=config.model.r)
    for name, paths in datasets.items()
}
for name, paths in datasets.items():
    print(f"{name:>5}: spot {paths.spot.shape}, shocks {paths.brownian_increments.shape}")

# %% [markdown]
# ## 2 - Sanity-check the simulator
# Before trusting any price we confirm the simulator is correct:
# * **(a)** the discounted price is a martingale: `E[e^{-r t_k} S_{t_k}] = S_0`;
# * **(b)** the Monte Carlo European call matches the closed-form Black-Scholes price.

# %%
test = datasets["test"]
discount = np.exp(-config.model.r * test.times)
martingale_dev = np.max(np.abs((discount * test.spot).mean(axis=0) - config.model.s0))
print(f"(a) max_k |E[e^(-r t_k) S_k] - S_0| = {martingale_dev:.4f}   (should be ~0)")

european = european_call(test, strike=config.strike, rate=config.model.r)
european_analytic = black_scholes_call_price(
    config.model.s0, config.strike, config.model.r, config.model.sigma, config.model.T
)
print(f"(b) European call: MC = {european.mean():.4f}   closed form = {european_analytic:.4f}")

# %% [markdown]
# ## 3 - The target payoff: arithmetic Asian call
# The payoff depends on the *average* price along the path. Here is the plain
# Monte Carlo price (high variance), and a look at a few simulated paths.

# %%
asian_plain = payoff["test"]
ci = 1.96 * asian_plain.std(ddof=1) / np.sqrt(asian_plain.size)
print(f"plain MC Asian price = {asian_plain.mean():.4f} +/- {ci:.4f}  (95% CI)")

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(20):
    ax.plot(test.times, test.spot[i], lw=0.8, alpha=0.8)
ax.set_title("20 simulated Black-Scholes paths")
ax.set_xlabel("time t")
ax.set_ylabel("asset price S_t")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"demo_paths_{config.n_steps}.png", dpi=130)
plt.show()

# %% [markdown]
# ## 4 - Asian-aware features
# A single price channel is a poor input (the signature of a 1-D path only sees
# its net move). We use three channels, all starting at 0:
# **time**, **log-price**, and the **running average** (the Asian state variable).
# These three are the *sufficient* state for the Asian call.

# %%
features = {name: asian_features(paths) for name, paths in datasets.items()}
print("channels:", features["train"].names)
print("feature path shape:", features["train"].values.shape, " (n_paths, N+1, d)")

# %% [markdown]
# ## 5 - Frozen randomized-signature reservoir (D = 64)
# The reservoir is an RNN with *frozen* random weights, driven by the feature
# increments: `h_{k+1} = h_k + sum_r rho(W_r h_k + xi_r) dx^{(r)}_{k+1}`. Because
# the control is linear in `beta`, each path collapses into a single regression row
#
# `Phi_i = sum_{k=0}^{N-1} phi_{i,k} * dW_{i,k+1}`,  with `phi_k = [1, t_k, h_k]`.

# %%
d_channels = features["train"].values.shape[2]   # read d from the features (= 3), not hardcoded
reservoir = RandomizedSignatureReservoir.draw(
    n_channels=d_channels, dim=RESERVOIR_DIM, rng=np.random.default_rng(config.seed + 100)
)

control_rows = {}      # Phi_i per dataset, shape (n, q)
for name, paths in datasets.items():
    control_rows[name] = assemble_control_features(
        reservoir, features[name], paths.times, paths.brownian_increments
    )
print("Phi shape:", control_rows["train"].shape, " (q = D + 2 =", control_rows["train"].shape[1], ")")

# %% [markdown]
# ## 6 - Train the readout (ridge) and show training results
# We fit `beta` on the **training** paths and choose the ridge penalty `lambda`
# by the variance reduction it gives on the **validation** paths. The training
# regression includes an intercept that absorbs the price level (so it does not
# distort the control) and is then dropped. Keeping train / val / test separate
# is what keeps the final estimator unbiased.

# %%
best_lambda, table = tune_ridge_lambda(
    control_rows["train"], payoff["train"],
    control_rows["val"], payoff["val"],
    RIDGE_LAMBDAS,
)
print(f"{'lambda':>12} {'train VR':>10} {'val VR':>10}")
for lam, train_vr_lam, val_vr_lam in table:
    flag = "  <- best" if lam == best_lambda else ""
    print(f"{lam:>12.1f} {train_vr_lam:>10.2f} {val_vr_lam:>10.2f}{flag}")

control = RidgeMartingaleControl.fit(control_rows["train"], payoff["train"], best_lambda)
train_vr = variance_reduction(control, control_rows["train"], payoff["train"])
print(f"\nchosen lambda      = {best_lambda:g}")
print(f"in-sample train VR = {train_vr:.2f}x   (R^2 = {1 - 1 / train_vr:.4f})")

# %%
lambdas, train_vrs, val_vrs = zip(*table)
positions = np.arange(len(lambdas))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(positions, train_vrs, "o-", label="train VR")
ax.plot(positions, val_vrs, "s-", label="validation VR")
ax.axvline(list(lambdas).index(best_lambda), color="k", ls="--", alpha=0.5, label="chosen lambda")
ax.set_xticks(positions)
ax.set_xticklabels([f"{lam:g}" for lam in lambdas])
ax.set_xlabel("ridge penalty lambda")
ax.set_ylabel("variance reduction VR")
ax.set_title("Variance reduction vs ridge penalty")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"demo_lambda_{config.n_steps}.png", dpi=130)
plt.show()

# %% [markdown]
# ## 7 - Evaluate on independent test paths
# The headline metric is the **standard error of the price**, not a per-path
# error. We build `Y^MC = G` and `Y^CV = G - C`, and report the variance
# reduction, the two standard errors, and a **bias check** (`mean(C)` should be
# ~0). `p_ref` is the *geometric* Asian price: by AM-GM it is an **exact lower
# bound** for our arithmetic payoff, so both prices should sit a little *above*
# it (a positive `p - p_ref` is expected, not a pricing error).

# %%
price_ref = geometric_asian_call_price(
    config.model.s0, config.strike, config.model.r,
    config.model.sigma, config.model.T, config.n_steps,
)
report = evaluate_control(
    control, control_rows["test"], payoff["test"],
    price_ref=price_ref,
    price_ref_label="geometric Asian (exact lower bound; arithmetic >= geometric)",
)
print(report.summary())

# %%
y_mc = payoff["test"]
y_cv = payoff["test"] - control.control(control_rows["test"])

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_left.hist(y_mc, bins=80, range=(-5, 30), alpha=0.55, density=True, label="Y^MC = G")
ax_left.hist(y_cv, bins=80, range=(-5, 30), alpha=0.55, density=True, label="Y^CV = G - C")
ax_left.axvline(report.price_cv, color="k", ls="--", lw=1, label="price")
ax_left.set_title("Controlled samples are concentrated")
ax_left.set_xlabel("sample value")
ax_left.set_ylabel("density")
ax_left.legend()

grid = np.unique(np.logspace(2, np.log10(y_mc.size), 40).astype(int))
se_mc = [y_mc[:n].std(ddof=1) / np.sqrt(n) for n in grid]
se_cv = [y_cv[:n].std(ddof=1) / np.sqrt(n) for n in grid]
ax_right.loglog(grid, se_mc, "o-", ms=3, label="plain Monte Carlo")
ax_right.loglog(grid, se_cv, "o-", ms=3, label="control variate")
ax_right.set_title(f"Price standard error  (VR = {report.variance_reduction:.0f}x)")
ax_right.set_xlabel("number of paths n")
ax_right.set_ylabel("standard error of the price")
ax_right.grid(True, which="both", alpha=0.3)
ax_right.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"demo_evaluation_{config.n_steps}.png", dpi=130)
plt.show()

# %% [markdown]
# # Part B - benchmark: reservoir vs the *real* truncated signature
#
# The reservoir is meant to be a cheap random surrogate for the finite truncated
# signature `s_k = Sig^{<=L}(x_0:k)`, computed via Chen's identity. We now run the
# signature through the **same** readout / ridge fit / evaluation - only the
# feature map changes - and compare. Everything below reuses the data, the
# payoff, and one shared helper, so there is no duplicated pipeline.

# %%
# One helper used for every feature map (reservoir or signature, any d, any size).
def run_control(assemble, train, val, test, lambdas=RIDGE_LAMBDAS):
    """assemble: (PathFeatures, times, dM) -> Phi.  Returns a metrics dict.

    ``train / val / test`` are bundles (PathFeatures, times, dM, payoff).
    """
    def phi_of(bundle):
        feats, times, shocks, _ = bundle
        return assemble(feats, times, shocks)

    phi_train = phi_of(train)
    phi_val = phi_of(val)
    start = time.perf_counter()
    phi_test = phi_of(test)                              # time the per-eval feature cost
    assemble_test_s = time.perf_counter() - start

    start = time.perf_counter()
    best_lam, _ = tune_ridge_lambda(phi_train, train[3], phi_val, val[3], lambdas)
    fitted = RidgeMartingaleControl.fit(phi_train, train[3], best_lam)
    fit_s = time.perf_counter() - start

    rep = evaluate_control(fitted, phi_test, test[3])
    return {
        "q": phi_test.shape[1],
        "vr": rep.variance_reduction,
        "var_drop": rep.var_mc - rep.var_cv,
        "var_cv": rep.var_cv,                            # residual variance Var(G - C)
        "se_cv": rep.se_cv,
        "control_mean": rep.control_mean,
        "assemble_test_s": assemble_test_s,
        "fit_s": fit_s,
        "best_lambda": best_lam,
    }


def reservoir_assembler(dim, n_channels, seed):
    res = RandomizedSignatureReservoir.draw(
        n_channels=n_channels, dim=dim, rng=np.random.default_rng(seed)
    )
    return lambda feats, times, shocks: assemble_control_features(res, feats, times, shocks)


def signature_assembler(level):
    return lambda feats, times, shocks: assemble_signature_control_features(
        feats, times, shocks, level=level
    )


def make_bundles(feature_map, path_sets, payoff_sets, cap=None):
    """Build train/val/test bundles from a feature dict + matching paths/payoffs.

    ``path_sets`` and ``payoff_sets`` are dicts keyed by split; this keeps the
    helper model-agnostic, so the same code drives Black-Scholes (scalar shocks
    ``(n, N)``) and Heston (vector shocks ``(n, N, 2)``) unchanged.
    """
    bundles = {}
    for name in ("train", "val", "test"):
        paths = path_sets[name]
        feats = feature_map[name]
        m = feats.values.shape[0] if cap is None else min(cap, feats.values.shape[0])
        bundles[name] = (
            PathFeatures(feats.values[:m], feats.names),
            paths.times,
            paths.brownian_increments[:m],          # (m, N) or (m, N, 2)
            payoff_sets[name][:m],
        )
    return bundles


def run_sweep(feature_map, path_sets, payoff_sets, reservoir_dims, signature_levels,
              cap=None, seed=200):
    """Run reservoir(D) and signature(L) controls on one feature map; return rows.

    The channel count d is read from the feature map, so it can never disagree
    with the actual features (no hardcoded 3 / 5). The martingale dimension m
    (1 for Black-Scholes, 2 for Heston) is inferred downstream from the shocks.
    """
    n_channels = feature_map["train"].values.shape[2]
    bundles = make_bundles(feature_map, path_sets, payoff_sets, cap=cap)
    rows = []
    for dim in reservoir_dims:
        label = f"reservoir D={dim}"
        print(f"  {label} ...", flush=True)
        metrics = run_control(
            reservoir_assembler(dim, n_channels, seed + dim),
            bundles["train"], bundles["val"], bundles["test"],
        )
        rows.append({"kind": "reservoir", "label": label, "feature_dim": dim, **metrics})
    for level in signature_levels:
        label = f"signature L={level}"
        print(f"  {label} ...", flush=True)
        metrics = run_control(
            signature_assembler(level),
            bundles["train"], bundles["val"], bundles["test"],
        )
        rows.append(
            {"kind": "signature", "label": label,
             "feature_dim": signature_dim(n_channels, level), **metrics}
        )
    return rows


def print_table(rows, n_eval):
    # resid/best = residual variance Var(G-C) relative to the best (lowest) here.
    # Unlike "retention", this stays discriminating even when every VR is high.
    best_resid = min(r["var_cv"] for r in rows)
    header = (f"{'method':<16}{'dim':>5}{'VR':>9}{'SE x':>7}"
              f"{'resid/best':>11}{'feat(s)':>9}{'fit(s)':>8}")
    print("-" * len(header))
    print(f"evaluation paths = {n_eval:,}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<16}{r['feature_dim']:>5}{r['vr']:>8.1f}x{np.sqrt(r['vr']):>6.1f}x"
            f"{r['var_cv'] / best_resid:>10.2f}x{r['assemble_test_s']:>8.2f}s{r['fit_s']:>7.2f}s"
        )
    print("-" * len(header))

# %% [markdown]
# ## 8 - Reservoir vs signature on the d = 3 Asian features
# Same readout, same data - only `h_k` vs `s_k` differs. Columns: `VR` / `SE x`
# are the headline; `resid/best` is the residual variance `Var(G-C)` relative to
# the best method here (1.00 = best, higher = more leftover variance - this stays
# discriminating even when every VR is large). Time is split into feature build
# (`feat`) and readout fit (`fit`); evaluation itself is one dot product.

# %%
# Both sweeps (d=3 here, d=5 below) use the SAME N_BENCH test paths and the same
# 20k train set, so every method is judged on the same ground and the feat / fit
# times are comparable across d.
print(f"d = 3 sweep on {N_BENCH:,} test paths:")
rows_d3 = run_sweep(
    features, datasets, payoff,
    reservoir_dims=SWEEP_DIMS, signature_levels=SWEEP_LEVELS, cap=N_BENCH,
)
print_table(rows_d3, N_BENCH)

# %% [markdown]
# ## 9 - When does the reservoir actually *save*? The `q_L = d^L` explosion
# The signature dimension grows like `d^L` in the number of channels `d`; the
# reservoir dimension `D` is fixed. With only `d = 3` channels the signature is
# small, so it is cheap and competitive. The saving becomes real when the path
# representation is richer (larger `d`).

# %%
fig, ax = plt.subplots(figsize=(8, 4.6))
d_grid = np.arange(2, 9)
for level in SWEEP_LEVELS:
    dims = [signature_dim(d, level) for d in d_grid]
    ax.plot(d_grid, dims, "o-", label=f"signature L={level}")
for dim in SWEEP_DIMS:
    ax.axhline(dim, ls="--", alpha=0.6, label=f"reservoir D={dim} (fixed)")
ax.set_yscale("log")
ax.set_xlabel("number of path channels  d")
ax.set_ylabel("feature dimension")
ax.set_title("Signature dimension q_L = d + ... + d^L  explodes in d;  reservoir D is fixed")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"demo_dimension_growth_{config.n_steps}.png", dpi=130)
plt.show()

# %% [markdown]
# ## 10 - Empirical higher-d run (d = 5 realistic channels)
# We add two genuine path-state channels - running **max** and **min** of
# log-price (lookback / barrier state) - to reach `d = 5`. Now signature
# dimensions are `L=2 -> 30`, `L=3 -> 155`, `L=4 -> 780`, while the reservoir
# stays at `D`. The richer representation also lifts VR for both methods, and
# `L = 4` (780-dim) is by far the heaviest config - this is the regime where the
# reservoir's fixed-`D` compression starts to pay off. This sweep uses the **same
# `N_BENCH` test paths and the same train set** as the d=3 sweep, so the times
# here are directly comparable to cell 8.

# %%
features_d5 = {
    name: asian_features(paths, include_running_max=True, include_running_min=True)
    for name, paths in datasets.items()
}
print("d = 5 channels:", features_d5["train"].names)
print(f"d = 5 sweep on {N_BENCH:,} test paths (same ground as d = 3):")
rows_d5 = run_sweep(
    features_d5, datasets, payoff,
    reservoir_dims=SWEEP_DIMS, signature_levels=SWEEP_LEVELS,
    cap=N_BENCH, seed=400,
)
print_table(rows_d5, N_BENCH)

# %%
fig, (ax_dim, ax_time) = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, xkey, xlabel in [
    (ax_dim, "feature_dim", "feature dimension  (D or q_L)"),
    (ax_time, "assemble_test_s", "feature-build time on test (s)"),
]:
    for rows, suffix, fmt in [(rows_d3, "d=3", "-"), (rows_d5, "d=5", "--")]:
        for kind, color in [("reservoir", "tab:blue"), ("signature", "tab:red")]:
            group = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r[xkey])
            ax.plot(
                [r[xkey] for r in group], [r["vr"] for r in group],
                fmt, marker="o" if kind == "reservoir" else "s", color=color, alpha=0.85,
                label=f"{kind} ({suffix})",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("variance reduction VR")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
ax_dim.set_xscale("log")
ax_dim.set_title("VR vs feature dimension")
ax_time.set_title("VR vs compute cost")
fig.suptitle("Reservoir vs finite truncated signature (d = 3 and d = 5)", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"demo_benchmark_{config.n_steps}.png", dpi=130)
plt.show()

# %% [markdown]
# ## Summary

# %%
# Data-driven so the narrative can never drift from the numbers above.
top_sig_d5 = max((r for r in rows_d5 if r["kind"] == "signature"),
                 key=lambda r: r["feature_dim"])           # richest = most expensive signature
best_res_d5 = max((r for r in rows_d5 if r["kind"] == "reservoir"),
                  key=lambda r: r["vr"])                    # strongest reservoir
time_speedup = top_sig_d5["assemble_test_s"] / best_res_d5["assemble_test_s"]
dim_factor = top_sig_d5["feature_dim"] / best_res_d5["feature_dim"]
resid_excess = best_res_d5["var_cv"] / top_sig_d5["var_cv"] - 1.0   # extra residual variance

print("=" * 70)
print("RESULT (Black-Scholes) - reservoir vs finite truncated signature")
print("=" * 70)
print(f"  Part A (reservoir D=64, d=3, {config.n_test:,} test paths):")
print(f"     VR = {report.variance_reduction:.1f}x, SE reduction = {report.se_reduction:.1f}x, "
      f"unbiased: mean(C) = {report.control_mean:+.4f}")
print("  Part B - reservoir vs the real signature:")
print("    * d=3: low-order signatures are cheap AND competitive; the reservoir")
print("      matches their VR but does not beat them on time (the exact signature")
print("      is tiny - 12 to 120 dims).")
print(f"    * d=5: {top_sig_d5['label']} is the richest at {top_sig_d5['feature_dim']}-dim, "
      f"VR {top_sig_d5['vr']:.1f}x @ {top_sig_d5['assemble_test_s']:.1f}s (heaviest).")
print(f"      {best_res_d5['label']} ({best_res_d5['feature_dim']}-dim) reaches "
      f"VR {best_res_d5['vr']:.1f}x @ {best_res_d5['assemble_test_s']:.1f}s:")
resid_phrase = (f"only ~{resid_excess * 100:.0f}% more residual variance" if resid_excess >= 0
                else f"even ~{-resid_excess * 100:.0f}% less residual variance")
print(f"      ~{time_speedup:.1f}x faster, ~{dim_factor:.0f}x fewer features, {resid_phrase} "
      f"-> compression starts to show.")
print("    * Signature dim grows as d^L (cell 9); the reservoir stays fixed at D.")
print(f"  Both sweeps share the same {N_BENCH:,}-path test set and train set, so the")
print("  feat / fit times above are directly comparable across d = 3 and d = 5.")
print("  Caveat: single random seed - a pilot that confirms the pipeline, not a")
print("  final benchmark.")
print("=" * 70)

# %% [markdown]
# # Part C - Heston: a vector martingale control (two Brownian shocks)
#
# Black-Scholes has a single Brownian driver, so the control integrates against a
# **scalar** increment. Heston adds a stochastic-variance factor with its *own*
# Brownian motion (correlated, `rho < 0` = leverage), so the martingale increment
# is a **vector** `dM = (dW^S, dW^v)` in R^2. The control becomes
#
# `C = sum_k Z_k . dM_{k+1}`,  one predictable readout direction per component,
#
# and the collapsed regression row concatenates one `[1, t_k, h_k]`-block per
# direction (so the reservoir row is `q = 2 * (D + 2)`). Everything else is
# identical: frozen reservoir, linear ridge readout, sample-split, and the same
# reservoir-vs-signature benchmark. The simulator uses full-truncation Euler, so
# this prices the **discretised** Heston target, not the exact continuous price.

# %%
heston_model = HestonModel()              # s0=100, v0=theta=0.04, kappa=1.5, xi=0.3, rho=-0.7
HESTON_SIZES = {"train": 20_000, "val": 5_000, "test": 40_000}
heston_datasets = make_heston_datasets(heston_model, HESTON_SIZES, config.n_steps, seed=0)
heston_payoff = {
    name: arithmetic_asian_call(paths, strike=config.strike, rate=heston_model.r)
    for name, paths in heston_datasets.items()
}
print("Heston model:", heston_model)
for name, paths in heston_datasets.items():
    print(f"{name:>5}: spot {paths.spot.shape}, variance {paths.variance.shape}, "
          f"dW {paths.brownian_increments.shape}")

# %% [markdown]
# ## C1 - Sanity-check the Heston simulator
# * discounted spot is a martingale: `E[e^{-r t_k} S_k] = S_0`;
# * variance stays non-negative (full truncation) and mean-reverts towards `theta`;
# * the realised correlation of the two Brownian increments matches `rho`.

# %%
htest = heston_datasets["test"]
discount = np.exp(-heston_model.r * htest.times)
martingale_dev = np.max(np.abs((discount * htest.spot).mean(axis=0) - heston_model.s0))
dW = htest.brownian_increments
realised_rho = np.corrcoef(dW[:, :, 0].ravel(), dW[:, :, 1].ravel())[0, 1]
print(f"(a) max_k |E[e^(-r t_k) S_k] - S_0| = {martingale_dev:.4f}   (should be ~0)")
print(f"(b) min variance = {htest.variance.min():.4f} (>= 0);  "
      f"mean v_T = {htest.variance[:, -1].mean():.4f}  (theta = {heston_model.theta})")
print(f"(c) realised corr(dW^S, dW^v) = {realised_rho:+.3f}  (target rho = {heston_model.rho})")
hci = 1.96 * heston_payoff["test"].std(ddof=1) / np.sqrt(heston_payoff["test"].size)
print(f"plain MC Heston Asian price = {heston_payoff['test'].mean():.4f} +/- {hci:.4f}  (95% CI)")

# %%
fig, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(12, 4.2))
for i in range(20):
    ax_s.plot(htest.times, htest.spot[i], lw=0.8, alpha=0.8)
    ax_v.plot(htest.times, htest.variance[i], lw=0.8, alpha=0.8)
ax_s.set_title("Heston spot paths"); ax_s.set_xlabel("t"); ax_s.set_ylabel("S_t")
ax_v.axhline(heston_model.theta, color="k", ls="--", lw=1, label="theta")
ax_v.set_title("Heston variance paths"); ax_v.set_xlabel("t"); ax_v.set_ylabel("v_t"); ax_v.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"demo_heston_paths_{config.n_steps}.png", dpi=130)
plt.show()

# %% [markdown]
# ## C2 - Heston features (d = 4) and the vector control headline
# Adapted channels: time, log-price, **variance** (the second state factor),
# running average. A single frozen reservoir (D = 64) + ridge readout, with the
# 2-D Brownian increments as the martingale control. The regression row has
# `q = 2 * (D + 2) = 132` columns: one `[1, t, h]`-block per Brownian direction.

# %%
heston_feats = {name: heston_features(paths) for name, paths in heston_datasets.items()}
print("Heston channels:", heston_feats["train"].names)

h_reservoir = RandomizedSignatureReservoir.draw(
    n_channels=heston_feats["train"].values.shape[2], dim=RESERVOIR_DIM,
    rng=np.random.default_rng(config.seed + 700),
)
h_phi = {
    name: assemble_control_features(
        h_reservoir, heston_feats[name], heston_datasets[name].times,
        heston_datasets[name].brownian_increments,                 # (n, N, 2) -> vector control
    )
    for name in ("train", "val", "test")
}
print("Heston Phi shape:", h_phi["train"].shape, " (q = 2 * (D + 2) =", h_phi["train"].shape[1], ")")

h_lambda, _ = tune_ridge_lambda(
    h_phi["train"], heston_payoff["train"], h_phi["val"], heston_payoff["val"], RIDGE_LAMBDAS
)
h_control = RidgeMartingaleControl.fit(h_phi["train"], heston_payoff["train"], h_lambda)
h_report = evaluate_control(h_control, h_phi["test"], heston_payoff["test"])
print(h_report.summary())

# %% [markdown]
# ## C3 - Heston benchmark: reservoir vs finite truncated signature
# Same vector control, only the path representation changes (`h_k` vs `s_k`). The
# signature dimensions for d = 4 are `L=2 -> 20`, `L=3 -> 84`, `L=4 -> 340` per
# Brownian block (doubled by the m = 2 directions in the regression row).

# %%
N_HESTON = 20_000
print(f"Heston (d = 4) sweep on {N_HESTON:,} test paths:")
rows_heston = run_sweep(
    heston_feats, heston_datasets, heston_payoff,
    reservoir_dims=SWEEP_DIMS, signature_levels=SWEEP_LEVELS, cap=N_HESTON, seed=700,
)
print_table(rows_heston, N_HESTON)

# %%
h_top_sig = max((r for r in rows_heston if r["kind"] == "signature"), key=lambda r: r["feature_dim"])
h_best_res = max((r for r in rows_heston if r["kind"] == "reservoir"), key=lambda r: r["vr"])
print("=" * 70)
print("RESULT (Heston) - vector martingale control, reservoir vs signature")
print("=" * 70)
print(f"  Headline (reservoir D={RESERVOIR_DIM}, d=4, vector m=2, {HESTON_SIZES['test']:,} test):")
print(f"     VR = {h_report.variance_reduction:.1f}x, SE reduction = {h_report.se_reduction:.1f}x, "
      f"unbiased: mean(C) = {h_report.control_mean:+.4f}")
print("  Benchmark: the vector control works for both representations -")
print(f"     {h_best_res['label']} ({h_best_res['feature_dim']}-dim) - best reservoir VR "
      f"{h_best_res['vr']:.1f}x @ {h_best_res['assemble_test_s']:.1f}s;")
print(f"     {h_top_sig['label']} ({h_top_sig['feature_dim']}-dim) - richest signature, VR "
      f"{h_top_sig['vr']:.1f}x @ {h_top_sig['assemble_test_s']:.1f}s.")
print("  Note: discretised full-truncation Euler target, single seed - a pilot.")
print("=" * 70)

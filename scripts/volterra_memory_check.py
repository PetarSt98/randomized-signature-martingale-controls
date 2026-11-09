"""Independent multi-seed check of the bounded-Volterra (H = 0.1) memory result.

Question (CAS slide): in the rough Volterra task the history maps forecast volatility better
than the static map yet hedge WORSE.  This script re-tests both halves on fresh dataset seeds
with the frozen CAS protocol (same channels, same readout [1, t, x, h], same ridge fit with
training-only RMS scaling, selection on validation over the declared candidate grid, three
fresh draws per selected setting, pooled test errors) and reports paired static-vs-history
differences on shared test paths.  The H = 1/2 variant (Markov in the observed state) is run
as a control: there, history should bring nothing.

  python scripts/volterra_memory_check.py --seeds 0,1,2,3,4,5,6,7,8,9 --tasks volterra,volterra_markov
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsmc.control import control_features  # noqa: E402
from rsmc.cost_hedging import hedge_readout_features  # noqa: E402
from rsmc.hedging import FrozenTanhFeatures, LeakyTanhFeatures, discounted_traded_increments, fit_fixed_premium_hedge  # noqa: E402
from rsmc.reservoir import RandomizedSignatureReservoir  # noqa: E402
from rsmc.transfer_tasks import make_transfer_dataset  # noqa: E402
from rsmc.volterra import BoundedVolterraModel, make_volterra_datasets  # noqa: E402

SIZES = {"train": 4000, "val": 2000, "test": 8000}
N_STEPS = 50
CANDIDATES = [dict(dim=64, weight_scale=s, bias_scale=1.0, lam=l) for s in (0.5, 1.0) for l in (0.0, 0.1, 10.0)]
AFFINE = [dict(lam=l) for l in (0.0, 0.1, 10.0)]
METHODS = ["affine", "static", "window4", "leaky", "reservoir"]
N_TUNING, N_FINAL = 3, 3
FORECAST_HORIZON = 5    # steps ahead over which future volatility is averaged for the forecast diagnostic


def feature_map(method, cand, rng, channels):
    if method == "affine":
        return None
    kw = dict(weight_scale=cand["weight_scale"], bias_scale=cand["bias_scale"])
    if method == "static":
        return FrozenTanhFeatures.draw(channels, cand["dim"], rng, window=0, **kw)
    if method == "window4":
        return FrozenTanhFeatures.draw(channels, cand["dim"], rng, window=4, **kw)
    if method == "leaky":
        return LeakyTanhFeatures.draw(channels, cand["dim"], rng, window=0, decay=0.8, **kw)
    if method == "reservoir":
        return RandomizedSignatureReservoir.draw(channels, cand["dim"], rng, **kw)
    raise ValueError(method)


def rows_for(data, times, rate, fmap):
    """phi (n, N+1, p) and gain rows psi (n, p) for one split."""
    phi = hedge_readout_features(data["features"], times, fmap)
    psi = control_features(phi, discounted_traded_increments(data["spot"], times, rate))
    return phi, psi


def errors(fitted, psi, payoff, premium):
    return payoff - premium - psi @ fitted.beta


def forecast_r2(phi_train, phi_test, target_train, target_test, lam=1e-3):
    """Ridge forecast of future volatility from phi_k (all k pooled); training-only RMS scaling."""
    Xtr = phi_train.reshape(-1, phi_train.shape[-1]); Xte = phi_test.reshape(-1, phi_test.shape[-1])
    ytr = target_train.reshape(-1); yte = target_test.reshape(-1)
    scale = np.sqrt(np.mean(Xtr ** 2, axis=0)); scale = np.where(scale < 1e-12, 1.0, scale)
    Ztr, Zte = Xtr / scale, Xte / scale
    G = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    theta = np.linalg.solve(G, Ztr.T @ ytr)
    pred = Zte @ theta
    return float(1.0 - np.mean((yte - pred) ** 2) / np.var(yte))


def future_vol_target(vol, horizon):
    """mean of sigma_{k+1..k+h} aligned with feature date k (k = 0..N-h); shape (n, N-h+1)."""
    n, n_pts = vol.shape
    out = np.empty((n, n_pts - horizon))
    for k in range(n_pts - horizon):
        out[:, k] = vol[:, k + 1:k + 1 + horizon].mean(axis=1)
    return out


def run_task(task, seeds, outdir, verbose=True):
    results = []
    for seed in seeds:
        t0 = time.time()
        data, times, rate, premium, meta = make_transfer_dataset(task, SIZES, N_STEPS, 2026110900 + seed)
        model = BoundedVolterraModel(hurst=0.1 if task == "volterra" else 0.5)
        paths = make_volterra_datasets(model, SIZES, N_STEPS, 2026110900 + seed)
        for split in SIZES:
            assert np.allclose(paths[split].spot, data[split]["spot"]), "volatility paths must align with the dataset"
        channels = data["train"]["features"].n_channels
        horizon = FORECAST_HORIZON
        ytr = future_vol_target(paths["train"].volatility, horizon)
        yte = future_vol_target(paths["test"].volatility, horizon)
        rec = {"task": task, "seed": seed, "premium": premium, "methods": {}}
        test_err = {}
        for mi, method in enumerate(METHODS):
            cands = AFFINE if method == "affine" else CANDIDATES
            # --- selection on validation (mean over tuning draws), never on test
            scores = []
            for ci, cand in enumerate(cands):
                vals = []
                for d in range(1 if method == "affine" else N_TUNING):
                    rng = np.random.default_rng(2026111000 + 10000 * seed + 1000 * mi + 100 * ci + d)
                    fmap = feature_map(method, cand, rng, channels)
                    _, psi_tr = rows_for(data["train"], times, rate, fmap)
                    _, psi_va = rows_for(data["val"], times, rate, fmap)
                    fitted = fit_fixed_premium_hedge(psi_tr, data["train"]["payoff"], premium, cand["lam"])
                    vals.append(float(np.mean(errors(fitted, psi_va, data["val"]["payoff"], premium) ** 2)))
                scores.append(float(np.mean(vals)))
            best = int(np.argmin(scores)); cand = cands[best]
            # --- final: fresh draws, pooled test errors; forecast diagnostic on the same draws
            errs, r2s = [], []
            for d in range(1 if method == "affine" else N_FINAL):
                rng = np.random.default_rng(2026112000 + 10000 * seed + 1000 * mi + d)
                fmap = feature_map(method, cand, rng, channels)
                phi_tr, psi_tr = rows_for(data["train"], times, rate, fmap)
                phi_te, psi_te = rows_for(data["test"], times, rate, fmap)
                fitted = fit_fixed_premium_hedge(psi_tr, data["train"]["payoff"], premium, cand["lam"])
                errs.append(errors(fitted, psi_te, data["test"]["payoff"], premium))
                r2s.append(forecast_r2(phi_tr[:, :ytr.shape[1]], phi_te[:, :yte.shape[1]], ytr, yte))
            errs = np.stack(errs)                          # (draws, n_test)
            test_err[method] = errs
            rec["methods"][method] = {
                "selected": cand, "validation_mse": scores[best], "candidate_scores": scores,
                "test_mse_pooled": float(np.mean(errs ** 2)), "test_rmse_pooled": float(np.sqrt(np.mean(errs ** 2))),
                "test_mse_per_draw": [float(np.mean(e ** 2)) for e in errs], "mean_error": float(errs.mean()),
                "forecast_r2_mean": float(np.mean(r2s)), "forecast_r2_per_draw": r2s,
            }
        # paired differences on the shared test paths (draw-averaged squared error per path)
        for method in METHODS:
            if method == "static":
                continue
            d = (test_err["static"] ** 2).mean(0) - (test_err[method] ** 2).mean(0)
            rec["methods"][method]["paired_vs_static"] = {
                "mse_diff_static_minus_method": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(d.size)),
                "rmse_ratio_method_over_static": float(np.sqrt(np.mean(test_err[method] ** 2) / np.mean(test_err["static"] ** 2)))}
        rec["seconds"] = time.time() - t0
        results.append(rec)
        if verbose:
            line = " | ".join(f"{m} {rec['methods'][m]['test_rmse_pooled']:.3f} (R2 {rec['methods'][m]['forecast_r2_mean']:.2f})" for m in METHODS)
            print(f"[{task}] seed {seed}: {line}  ({rec['seconds']:.0f}s)", flush=True)
        with open(f"{outdir}/{task}_seeds.json", "w") as f:
            json.dump(results, f, indent=1)
    return results


def summarize(results, task):
    seeds = [r["seed"] for r in results]
    lines = [f"### {task}: {len(seeds)} dataset seeds, 3 fresh draws each, 8000 test paths per seed",
             "", "| method | pooled test RMSE (mean ± sd over seeds) | RMSE ratio vs static (mean, range) | seeds method beats static | forecast R² of future vol (mean) |",
             "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        rm = np.array([r["methods"][m]["test_rmse_pooled"] for r in results])
        r2 = np.mean([r["methods"][m]["forecast_r2_mean"] for r in results])
        if m == "static":
            lines.append(f"| {m} | {rm.mean():.3f} ± {rm.std():.3f} | 1 | — | {r2:.3f} |")
        else:
            ratios = np.array([r["methods"][m]["paired_vs_static"]["rmse_ratio_method_over_static"] for r in results])
            wins = int(sum(r["methods"][m]["paired_vs_static"]["mse_diff_static_minus_method"] > 0 for r in results))
            lines.append(f"| {m} | {rm.mean():.3f} ± {rm.std():.3f} | {ratios.mean():.3f} [{ratios.min():.3f}, {ratios.max():.3f}] | {wins}/{len(results)} | {r2:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tasks", default="volterra,volterra_markov")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "volterra_check"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    md = []
    for task in args.tasks.split(","):
        res = run_task(task, seeds, args.outdir)
        md.append(summarize(res, task))
    text = "\n\n".join(md)
    print("\n" + text)
    with open(f"{args.outdir}/SUMMARY.md", "w") as f:
        f.write(text + "\n")

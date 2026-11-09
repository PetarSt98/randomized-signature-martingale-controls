"""Mechanism diagnostics.

(1) Linear decodability of the hidden liquidity regime (AUC) and of the pending impact (R^2) from each
    frozen representation along a fixed behaviour policy (damped delta hedge) -- what each feature map
    'knows' before any policy training.
(2) Behaviour of trained policies by regime: average trade size per step in the liquid vs illiquid regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsmc.impact_hedging import (EWMABank, HALF_LIVES as HL, LinearReadoutPolicy, Reservoir, StaticTanh,  # noqa: E402
                                 bachelier_delta, make_policy, market_from_args, rollout)

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
       "green": "#008300", "violet": "#6250d6", "red": "#e34948", "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e"}


def decodability(mkt, outdir, tag, ks=(5, 10, 20, 40, 60, 80, 100, 119), n=6000, seed=0):
    p0 = mkt.premium
    reps = {
        "static_tanh": lambda g: StaticTanh(256, g),
        "ewma_tanh": lambda g: EWMABank(HL, g, D=256),
        "cross_tanh": lambda g: EWMABank(HL, g, D=256, extra_channels=True),
        "reservoir": lambda g: Reservoir(128, g),
        "reservoir_tanh": lambda g: Reservoir(128, g, tanh_head=128),
    }
    tr = mkt.sample(n, torch.Generator().manual_seed(11 + seed)); te = mkt.sample(n, torch.Generator().manual_seed(12 + seed))

    def pos(k, obs):  # damped delta hedge on the observed price: a generic behaviour policy
        tau = mkt.T - k * mkt.dt
        return obs["Q_prev"] + 0.5 * (bachelier_delta(obs["s"] + mkt.F0, mkt.K, mkt.sigma, tau) - obs["Q_prev"])

    res = {}
    for name, mk in reps.items():
        fm = mk(torch.Generator().manual_seed(3 + seed)); pol = LinearReadoutPolicy(fm)
        with torch.no_grad():
            o_tr = rollout(pol, mkt, tr, p0, position=pos, record=True); o_te = rollout(pol, mkt, te, p0, position=pos, record=True)
        aucs, r2s = [], []
        for k in ks:
            Xtr = o_tr["rec"]["phi"][:, k].numpy(); Xte = o_te["rec"]["phi"][:, k].numpy()
            mu = Xtr.mean(0); sd = Xtr.std(0); sc = np.where(sd < 1e-6, 1.0, sd); Xtr = (Xtr - mu) / sc; Xte = (Xte - mu) / sc
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, tr["z"].numpy())
            aucs.append(float(roc_auc_score(te["z"].numpy(), clf.decision_function(Xte))))
            rg = Ridge(alpha=1.0).fit(Xtr, o_tr["rec"]["I"][:, k].numpy()); r2s.append(float(rg.score(Xte, o_te["rec"]["I"][:, k].numpy())))
        res[name] = {"k": list(ks), "auc": aucs, "r2_I": r2s}
        print(name, "AUC", [round(a, 3) for a in aucs], "R2(I)", [round(r, 2) for r in r2s], flush=True)
    with open(f"{outdir}/{tag}_decodability.json", "w") as f:
        json.dump(res, f, indent=1)
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=160)
    cols = {"static_tanh": PAL["gray"], "ewma_tanh": PAL["orange"], "cross_tanh": PAL["yellow"], "reservoir": PAL["blue"], "reservoir_tanh": PAL["violet"]}
    labels = {"static_tanh": "static tanh", "ewma_tanh": "EWMA bank + tanh", "cross_tanh": "hand-crafted propagator stats", "reservoir": "reservoir", "reservoir_tanh": "reservoir + tanh head"}
    for name, r in res.items():
        axes[0].plot(r["k"], r["auc"], marker="o", ms=3.5, lw=1.6, color=cols[name], label=labels[name])
        axes[1].plot(r["k"], r["r2_I"], marker="o", ms=3.5, lw=1.6, color=cols[name], label=labels[name])
    axes[0].set_title("Hidden liquidity regime: linear decodability (AUC)", fontsize=9.5, loc="left")
    axes[1].set_title("Pending own impact $I_k$: linear $R^2$", fontsize=9.5, loc="left")
    for ax in axes:
        ax.set_xlabel("rebalancing step k"); ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].axhline(0.5, color="#c9c8c3", lw=0.8, ls="--"); axes[0].set_ylim(0.45, 1.0)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_fig_decodability.png"); plt.close(fig)
    return res


def behaviour(mkt, outdir, tag, seed, names=("static_tanh", "ewma_tanh", "reservoir_tanh", "oracle_state"), n=6000):
    """Average absolute trade per step by regime for trained policies (needs saved *_policy.pt)."""
    p0 = mkt.premium
    te = mkt.sample(n, torch.Generator().manual_seed(9000 + seed))
    out_all = {}
    for name in names:
        f = f"{outdir}/{tag}_{name}_s{seed}_policy.pt"
        if not os.path.exists(f):
            print("missing", f); continue
        st = torch.load(f)
        fm = make_policy(name, 256, torch.Generator().manual_seed(2000 + seed))
        pol = LinearReadoutPolicy(fm, st["w"]); pol.scale = st["scale"]
        with torch.no_grad():
            o = rollout(pol, mkt, te, p0, record=True)
        Q = o["rec"]["Q"]; q = torch.diff(Q, dim=1).abs()
        z = te["z"]
        out_all[name] = {"liquid": q[z == 0].mean(0).tolist(), "illiquid": q[z == 1].mean(0).tolist(),
                         "rmse": float(o["err"].pow(2).mean().sqrt())}
        print(name, "rmse", round(out_all[name]["rmse"], 4), "mean |q| liquid", round(float(q[z == 0].mean()), 4), "illiquid", round(float(q[z == 1].mean()), 4))
    with open(f"{outdir}/{tag}_behaviour_s{seed}.json", "w") as fh:
        json.dump(out_all, fh)
    if out_all:
        fig, axes = plt.subplots(1, len(out_all), figsize=(2.6 * len(out_all) + 1, 3.2), dpi=160, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (name, r) in zip(axes, out_all.items()):
            ax.plot(r["liquid"], color=PAL["aqua"], lw=1.6, label="liquid regime")
            ax.plot(r["illiquid"], color=PAL["orange"], lw=1.6, label="illiquid regime")
            ax.set_title(name, fontsize=9); ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.set_xlabel("step")
        axes[0].set_ylabel("mean |trade| per step"); axes[0].legend(frameon=False, fontsize=7.5)
        fig.suptitle("Trained policies: how much they trade, by (hidden) regime", fontsize=10, x=0.02, ha="left")
        fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_fig_behaviour_s{seed}.png"); plt.close(fig)
    return out_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "reports" / "impact_hedging" / "results")); ap.add_argument("--tag", default="main")
    ap.add_argument("--kappa", default="0.5,20"); ap.add_argument("--lam", default="0.71,0.95"); ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--what", default="decode,behaviour"); ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    mkt = market_from_args(args.kappa, args.lam, args.steps)
    if "decode" in args.what:
        decodability(mkt, args.outdir, args.tag)
    if "behaviour" in args.what:
        behaviour(mkt, args.outdir, args.tag, args.seed)

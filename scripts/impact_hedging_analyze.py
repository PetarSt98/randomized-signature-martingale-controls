"""Aggregate impact-hedging results: markdown tables, paired bootstrap CIs, figures, sensitivity table."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
       "green": "#008300", "violet": "#6250d6", "red": "#e34948", "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e"}

LABEL = {
    "static_tanh": "Static tanh (memoryless)",
    "ewma_linear": "EWMA bank, linear readout",
    "ewma_tanh": "EWMA bank + tanh head",
    "cross_tanh": "Hand-crafted propagator stats + tanh",
    "reservoir": "Reservoir (Petar's readout)",
    "reservoir_tanh": "Reservoir + tanh head",
    "reservoir_norec": "Reservoir, recurrence removed",
    "oracle_state": "Oracle (sees F, impact, regime)",
    "gru": "GRU trained end-to-end (same budget)",
}
SHORT = {
    "static_tanh": "Static tanh\n(no memory)", "ewma_linear": "EWMA bank\nlinear readout", "ewma_tanh": "EWMA bank\n+ tanh head",
    "cross_tanh": "Hand-crafted\npropagator stats", "reservoir_norec": "Reservoir\nno recurrence", "reservoir": "Reservoir\n(Petar's readout)",
    "reservoir_tanh": "Reservoir\n+ tanh head", "gru": "GRU\n(same budget)", "oracle_state": "Oracle\n(F, impact, regime)",
}
ORDER = ["static_tanh", "ewma_linear", "ewma_tanh", "cross_tanh", "reservoir_norec", "reservoir", "reservoir_tanh", "gru", "oracle_state"]


def load(outdir, tag):
    rows = []
    for f in sorted(glob.glob(f"{outdir}/{tag}_*_s*.json")):
        with open(f) as fh:
            row = json.load(fh)
        if isinstance(row, dict) and "policy" in row:      # skip diagnostics files that share the tag prefix
            rows.append(row)
    return rows


def paired_bootstrap(outdir, tag, a, b, seeds, n_boot=2000, rng=None):
    """Paired bootstrap (over test paths, pooled over seeds) of MSE(a) - MSE(b) and RMSE ratio."""
    rng = rng or np.random.default_rng(0)
    ea, eb = [], []
    for s in seeds:
        fa, fb = f"{outdir}/{tag}_{a}_s{s}_err.npy", f"{outdir}/{tag}_{b}_s{s}_err.npy"
        if os.path.exists(fa) and os.path.exists(fb):
            ea.append(np.load(fa)); eb.append(np.load(fb))
    if not ea:
        return None
    ea, eb = np.concatenate(ea), np.concatenate(eb)
    n = ea.size
    d = ea ** 2 - eb ** 2
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.sqrt(np.mean(ea[idx] ** 2)) / np.sqrt(np.mean(eb[idx] ** 2)))
    boots = np.array(boots)
    return {"rmse_ratio": float(np.sqrt(np.mean(ea ** 2)) / np.sqrt(np.mean(eb ** 2))),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "mse_diff_mean": float(d.mean()), "mse_diff_se": float(d.std(ddof=1) / np.sqrt(n)), "n": int(n)}


def table(rows, outdir, tag):
    names = [n for n in ORDER if any(r["policy"] == n for r in rows)]
    seeds = sorted(set(r["seed"] for r in rows))
    lines = ["| Hedger | features | test net RMSE (mean ± sd over seeds) | liquid regime | illiquid regime | mean cost | CVaR95 | vs. reservoir+tanh (RMSE ratio, 95% CI) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for n in names:
        rs = [r for r in rows if r["policy"] == n]
        rm = np.array([r["net_rmse"] for r in rs]); z0 = np.array([r["net_rmse_z0"] for r in rs]); z1 = np.array([r["net_rmse_z1"] for r in rs])
        cost = np.mean([r["mean_cost"] for r in rs]); cvar = np.mean([r["cvar95"] for r in rs])
        pb = paired_bootstrap(outdir, tag, n, "reservoir_tanh", seeds) if n != "reservoir_tanh" else None
        cmp_ = f"{pb['rmse_ratio']:.3f} [{pb['ci95'][0]:.3f}, {pb['ci95'][1]:.3f}]" if pb else "—"
        lines.append(f"| {LABEL[n]} | {rs[0]['dim']} | **{rm.mean():.3f}** ± {rm.std():.3f} | {z0.mean():.3f} | {z1.mean():.3f} | {cost:.3f} | {cvar:.2f} | {cmp_} |")
        summary[n] = {"rmse_mean": float(rm.mean()), "rmse_sd": float(rm.std()), "z0": float(z0.mean()), "z1": float(z1.mean()),
                      "cost": float(cost), "cvar95": float(cvar), "n_seeds": len(rs), "dim": rs[0]["dim"], "vs_reservoir_tanh": pb}
    return "\n".join(lines), summary, names, seeds


def fig_main(rows, names, outdir, tag):
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    means, sds, z0s, z1s = [], [], [], []
    for n in names:
        rs = [r for r in rows if r["policy"] == n]
        means.append(np.mean([r["net_rmse"] for r in rs])); sds.append(np.std([r["net_rmse"] for r in rs]))
        z0s.append(np.mean([r["net_rmse_z0"] for r in rs])); z1s.append(np.mean([r["net_rmse_z1"] for r in rs]))
    x = np.arange(len(names))
    w = 0.26
    cols = {"all": PAL["blue"], "liq": PAL["aqua"], "ill": PAL["orange"]}
    ax.bar(x - w, means, w, color=cols["all"], label="all paths", yerr=sds, capsize=2, error_kw={"lw": 0.8, "color": PAL["ink2"]})
    ax.bar(x, z0s, w, color=cols["liq"], label="liquid regime")
    ax.bar(x + w, z1s, w, color=cols["ill"], label="illiquid regime")
    for xi, m in zip(x, means):
        ax.text(xi - w, m + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color=PAL["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
    ax.set_ylabel("test net hedging RMSE (price units)", fontsize=9)
    ax.set_title("Hedging an ATM call under transient impact with hidden liquidity: lower is better", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{tag}_fig_rmse.png"); plt.close(fig)


def fig_history(rows, names, outdir, tag, seed=0):
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=160)
    palette = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["yellow"], PAL["magenta"], PAL["green"], PAL["violet"], PAL["red"], PAL["gray"]]
    for i, n in enumerate(names):
        rs = [r for r in rows if r["policy"] == n and r["seed"] == seed]
        if not rs:
            continue
        h = np.array(rs[0]["history"])
        ax.plot(h[:, 0], np.sqrt(h[:, 2]), lw=1.6, color=palette[i % len(palette)], label=LABEL[n])
    ax.set_xlabel("training iteration"); ax.set_ylabel("validation net RMSE")
    ax.set_ylim(top=min(ax.get_ylim()[1], 3.0))
    ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Validation curves (seed {seed})", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_fig_history.png"); plt.close(fig)




def sensitivity_table(outdir):
    """How the memory advantage scales with the hidden-regime separation (tags produced by impact_hedging_run.py)."""
    CFG = [("control_nohidden", "no hidden regime (κ=5 both, λ=0.9)"), ("sens_k5", "κ: 0.5 vs 5"), ("sens_k10", "κ: 0.5 vs 10"), ("main", "κ: 0.5 vs 20 (main)")]
    POL = ["static_tanh", "ewma_tanh", "cross_tanh", "reservoir_tanh", "oracle_state"]
    NAMES = {"static_tanh": "static tanh", "ewma_tanh": "EWMA+tanh", "cross_tanh": "hand-crafted", "reservoir_tanh": "reservoir+tanh", "oracle_state": "oracle"}

    def mean_rmse(tag, pol):
        fs = glob.glob(f"{outdir}/{tag}_{pol}_s*.json")
        if not fs:
            return None, 0
        v = [json.load(open(f))["net_rmse"] for f in fs]
        return float(np.mean(v)), len(v)

    lines = ["| hidden-regime separation | seeds | " + " | ".join(NAMES[p] for p in POL) + " | reservoir+tanh vs EWMA+tanh | reservoir+tanh vs static | gap to oracle closed |",
             "|---|---:|" + "---:|" * len(POL) + "---:|---:|---:|"]
    for tag, label in CFG:
        vals = {p: mean_rmse(tag, p) for p in POL}
        if vals["reservoir_tanh"][0] is None:
            continue
        n = max(v[1] for v in vals.values())
        r, e, s, o = vals["reservoir_tanh"][0], vals["ewma_tanh"][0], vals["static_tanh"][0], vals["oracle_state"][0]
        closed = (s - r) / (s - o) if s and o and s - o > 0.05 else None
        closed_s = f"{100*closed:.0f}%" if closed is not None else f"n/a (static–oracle gap only {s-o:.3f})"
        lines.append(f"| {label} | {n} | " + " | ".join(f"{vals[p][0]:.3f}" if vals[p][0] is not None else "—" for p in POL)
                     + f" | {100*(1 - r/e):+.1f}% | {100*(1 - r/s):+.1f}% | {closed_s} |")
    md = "\n".join(lines)
    open(f"{outdir}/sensitivity_table.md", "w").write(md + "\n")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "reports" / "impact_hedging" / "results")); ap.add_argument("--tag", default="main")
    ap.add_argument("--sens", action="store_true", help="also build the regime-separation sensitivity table")
    args = ap.parse_args()
    rows = load(args.outdir, args.tag)
    md, summary, names, seeds = table(rows, args.outdir, args.tag)
    print(f"tag={args.tag} seeds={seeds}")
    print(md)
    with open(f"{args.outdir}/{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(f"{args.outdir}/{args.tag}_table.md", "w") as f:
        f.write(md + "\n")
    fig_main(rows, names, args.outdir, args.tag)
    fig_history(rows, names, args.outdir, args.tag)
    if args.sens:
        print(sensitivity_table(args.outdir))

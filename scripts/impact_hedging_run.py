"""Hedging under transient impact with hidden liquidity: same budget for every hedger, several seeds,
common random numbers, per-path errors saved for paired bootstrap (see impact_hedging_analyze.py).

Design (all policies): frozen feature map -> features standardized by their warm-start RMS -> linear readout
trained by Adam (cosine schedule) on the mean squared net hedging error of closed-loop rollouts,
validation-selected checkpoint, evaluated on an independent test set shared by all policies.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import time

import numpy as np
import torch

from rsmc.impact_hedging import (LEARNING_RATE_OVERRIDE as LR, LinearReadoutPolicy, evaluate, make_policy,  # noqa: E402
                                 market_from_args, train, warm_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--kappa", type=str, default="0.5,20")
    ap.add_argument("--lam", type=str, default="0.71,0.95")
    ap.add_argument("--policies", type=str,
                    default="static_tanh,ewma_linear,ewma_tanh,cross_tanh,reservoir,reservoir_tanh,reservoir_norec,oracle_state,gru")
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--outdir", type=str, default=str(ROOT / "reports" / "impact_hedging" / "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mkt = market_from_args(args.kappa, args.lam, args.steps)
    p0 = mkt.premium
    print(f"[{args.tag}] steps={mkt.n_steps} premium={p0:.4f} regimes={[(r.kappa, r.lam) for r in mkt.regimes]}", flush=True)

    for seed in [int(s) for s in args.seeds.split(",")]:
        val_batch = mkt.sample(args.n_val, torch.Generator().manual_seed(5000 + seed))
        test_batch = mkt.sample(args.n_test, torch.Generator().manual_seed(9000 + seed))
        ws_batch = mkt.sample(2048, torch.Generator().manual_seed(7000 + seed))
        for name in args.policies.split(","):
            fpath = f"{args.outdir}/{args.tag}_{name}_s{seed}.json"
            if os.path.exists(fpath):
                print(f"  skip {fpath} (exists)")
                continue
            t0 = time.time()
            fm = make_policy(name, args.D, torch.Generator().manual_seed(2000 + seed))
            pol = LinearReadoutPolicy(fm)
            warm_start(pol, mkt, ws_batch, p0)
            gen_train = torch.Generator().manual_seed(1000 + seed)       # identical training paths for all policies
            tr = train(pol, mkt, p0, gen_train, n_iters=args.iters, batch_size=args.batch, lr=LR.get(name, args.lr),
                       val_batch=val_batch)
            m, out = evaluate(pol, mkt, p0, test_batch)
            m.update({"policy": name, "fmap": fm.name, "seed": seed, "dim": fm.dim, "best_val_mse": tr["best_val_mse"],
                      "train_time": tr["train_time"], "total_time": time.time() - t0, "lr": LR.get(name, args.lr),
                      "history": tr["history"], "args": vars(args), "premium": p0})
            np.save(f"{args.outdir}/{args.tag}_{name}_s{seed}_err.npy", out["err"].numpy())
            np.save(f"{args.outdir}/{args.tag}_{name}_s{seed}_z.npy", out["z"].numpy())
            np.save(f"{args.outdir}/{args.tag}_{name}_s{seed}_cost.npy", out["cost"].numpy())
            torch.save({"w": pol.w.detach(), "scale": pol.scale, "fmap_state": getattr(fm, "cell", None) and fm.cell.state_dict()},
                       f"{args.outdir}/{args.tag}_{name}_s{seed}_policy.pt")
            with open(fpath, "w") as f:
                json.dump(m, f)
            print(f"  seed {seed} {name:16s} dim={fm.dim:4d} test net_RMSE={m['net_rmse']:.4f} "
                  f"(liquid {m['net_rmse_z0']:.3f} | illiquid {m['net_rmse_z1']:.3f}) cost={m['mean_cost']:.3f} "
                  f"turn={m['mean_turnover']:.2f} cvar95={m['cvar95']:.3f} val={m['best_val_mse']:.4f} {m['total_time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

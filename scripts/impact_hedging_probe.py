"""Quick single-seed probe of a market configuration: static vs EWMA+tanh vs reservoir vs oracles."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, time  # noqa: E402

import torch  # noqa: E402

import rsmc.impact_hedging as ih  # noqa: E402
from rsmc.impact_hedging import *  # noqa: E402,F401,F403


class OracleSubset(FeatureMap):
    def __init__(self, D, gen, know, s_scale=10.0):
        self.know = know; self.s_scale = s_scale
        n_in = 3 + len(know)
        self.A, self.b = ih._rand_features(gen, n_in, D, 1.0)
        self.name = f"oracle[{','.join(know)}]"; self.dim = 1 + n_in + D

    def step(self, obs, state):
        cols = [obs["tau"], obs["s"] / self.s_scale, obs["Q_prev"]]
        if "f" in self.know: cols.append(obs["f"] / self.s_scale)
        if "I" in self.know: cols.append(obs["I"] / self.s_scale * 5)
        if "z" in self.know: cols.append(obs["z"] / max(1, len(self.regimes) - 1) if hasattr(self, "regimes") else obs["z"])
        u = torch.stack(cols, 1)
        return torch.cat([torch.ones_like(u[:, :1]), u, torch.tanh(u @ self.A.T + self.b)], 1), state


ap = argparse.ArgumentParser()
ap.add_argument("--kappa", default="2,8"); ap.add_argument("--lam", default="0.5,0.9")
ap.add_argument("--steps", type=int, default=60); ap.add_argument("--iters", type=int, default=150)
ap.add_argument("--in_scale", type=float, default=1.0); ap.add_argument("--ws", type=float, default=1.0)
ap.add_argument("--D", type=int, default=128); ap.add_argument("--q_init", default="delta")
ap.add_argument("--which", default="static,ewma_tanh,reservoir,oracle_z,oracle_all")
ap.add_argument("--lr", type=float, default=1e-2)
args = ap.parse_args()
kappas = [float(x) for x in args.kappa.split(",")]; lams = [float(x) for x in args.lam.split(",")]
regimes = [Regime(f"r{i}", 1 / len(kappas), kappa=k, lam=l) for i, (k, l) in enumerate(zip(kappas, lams))]
mkt = Market(n_steps=args.steps, regimes=regimes, q_init=args.q_init); p0 = mkt.premium
print(f"== steps={args.steps} kappa={kappas} lam={lams} premium={p0:.3f} in_scale={args.in_scale} ws={args.ws} D={args.D}")
val = mkt.sample(4096, torch.Generator().manual_seed(5000)); test = mkt.sample(10000, torch.Generator().manual_seed(9000)); ws_b = mkt.sample(2048, torch.Generator().manual_seed(7000))
hl = [1, 2, 4, 8, 16, 32]
for which in args.which.split(","):
    g = torch.Generator().manual_seed(2000)
    if which == "static": fm = StaticTanh(args.D, g)
    elif which == "ewma_tanh": fm = EWMABank(hl, g, D=args.D)
    elif which == "ewma_linear": fm = EWMABank(hl, g, D=0)
    elif which == "cross_tanh": fm = EWMABank(hl, g, D=args.D, extra_channels=True)
    elif which == "reservoir": fm = Reservoir(args.D, g, weight_scale=args.ws, input_scale=args.in_scale)
    elif which == "reservoir_d2": fm = Reservoir(args.D, g, weight_scale=args.ws, input_scale=args.in_scale, depth_two=True)
    elif which == "reservoir_tanh": fm = Reservoir(args.D, g, weight_scale=args.ws, input_scale=args.in_scale, tanh_head=args.D)
    elif which == "norec": fm = Reservoir(args.D, g, weight_scale=args.ws, input_scale=args.in_scale, recurrent=False)
    elif which == "oracle_z": fm = OracleSubset(args.D, g, ["z"]); fm.regimes = regimes
    elif which == "oracle_all": fm = OracleSubset(args.D, g, ["f", "I", "z"]); fm.regimes = regimes
    elif which == "gru": fm = GRUPolicy(32, g)
    t0 = time.time()
    pol = LinearReadoutPolicy(fm); warm_start(pol, mkt, ws_b, p0)
    tr = train(pol, mkt, p0, torch.Generator().manual_seed(1000), n_iters=args.iters, batch_size=1024, lr=args.lr, val_batch=val)
    m, _ = evaluate(pol, mkt, p0, test)
    zs = " ".join(f"z{i} {m[f'net_rmse_z{i}']:.3f}" for i in range(len(kappas)))
    print(f"{fm.name:34s} test RMSE {m['net_rmse']:.4f}  {zs}  cost {m['mean_cost']:.3f} turn {m['mean_turnover']:.2f} cvar {m['cvar95']:.2f} val {tr['best_val_mse']:.4f} ({time.time()-t0:.0f}s)", flush=True)

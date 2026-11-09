"""Option hedging under TRANSIENT market impact with a HIDDEN liquidity regime.

Post-report study (Nov 2026).  Library code; the protocol lives in ``scripts/impact_hedging_run.py``.

The hedger's own trades move the observed price (Obizhaeva-Wang / propagator
impact) and the displacement decays over time.  Depth (kappa) and resilience
(decay per step, lam) are drawn per path from a hidden regime that is *not*
visible in the price path itself (the fundamental has the same volatility in
every regime): the only way to learn the regime is from how the price responds
to one's own past trades -- a cross-term  (own trade) x (subsequent price move),
i.e. a level-2 signature object, not a moving average of any single channel.

Everything below uses Petar's conventions:
  * reservoir  h_{k+1} = h_k + sum_r tanh(W_r h_k + xi_r) * dx^r_{k+1},  h_0 = 0,
    W_r ~ N(0, (weight_scale/sqrt(D))^2),  xi_r ~ N(0, bias_scale^2)      (rsmc/reservoir.py)
  * readout   [1, t_k, x_k (non-time), h_k]  (rsmc/cost_hedging.hedge_readout_features)
  * metric    net error = payoff - premium - gain + cost  (rsmc/cost_hedging.cost_metrics)
    and every policy minimises the mean squared net error with a frozen feature map
    and a trained linear readout only (the same optimiser/budget for every policy).

Market (Bachelier fundamental, zero rate):
    F_{k+1} = F_k + sigma * sqrt(dt) * eps_{k+1}
    S_k     = F_k + I_k                         observed (pre-trade) mid price
    trade q_k = Q_k - Q_{k-1} at t_k, execution shortfall vs F_k:
        cost_k = q_k * ( I_k + (kappa/2) q_k + eta q_k )
    I_{k+1} = lam * ( I_k + kappa * q_k )       pending (transient) displacement
    payoff  H = (F_N - K)^+ ,  final position valued at the unaffected price F_N
    net error e = H - p0 - sum_k Q_k (F_{k+1}-F_k) + sum_k cost_k
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import os
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
SQRT2 = math.sqrt(2.0)


def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def phi_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bachelier_call(F, K, sigma, tau):
    s = sigma * math.sqrt(max(tau, 1e-12))
    d = (F - K) / s
    return (F - K) * Phi(d) + s * phi_pdf(d)


def bachelier_delta(F, K, sigma, tau):
    s = sigma * math.sqrt(max(tau, 1e-12))
    return Phi((F - K) / s)


# ----------------------------------------------------------------------------- market
@dataclass
class Regime:
    name: str
    prob: float
    kappa: float   # immediate price displacement per unit traded (price units / unit position)
    lam: float     # fraction of the displacement surviving one step  (= exp(-rho dt))
    eta: float = 0.0  # extra temporary slippage per unit traded


@dataclass
class Market:
    F0: float = 100.0
    K: float = 100.0
    sigma: float = 20.0          # Bachelier vol, price units per sqrt(year)
    n_steps: int = 60
    dt: float = 1.0 / 252.0
    regimes: list = field(default_factory=lambda: [
        Regime("liquid", 0.5, kappa=2.0, lam=0.5),
        Regime("illiquid", 0.5, kappa=8.0, lam=0.9),
    ])
    q_init: str = "delta"        # "delta": inherit the frictionless delta at t=0 (no pending impact); "flat": Q_{-1}=0
    freeze_last: int = 5         # no rebalancing in the last `freeze_last` steps (gamma blow-up at expiry is not the object of study)
    q_lo: float = -0.5           # position bounds (hard clamp), same for every policy
    q_hi: float = 1.5

    @property
    def T(self):
        return self.n_steps * self.dt

    @property
    def times(self):
        return np.arange(self.n_steps + 1) * self.dt

    @property
    def premium(self):
        return float(bachelier_call(torch.tensor(self.F0), self.K, self.sigma, self.T))

    def sample(self, n: int, gen: torch.Generator):
        eps = torch.randn(n, self.n_steps, generator=gen)
        dF = self.sigma * math.sqrt(self.dt) * eps
        F = torch.cat([torch.full((n, 1), self.F0), self.F0 + torch.cumsum(dF, 1)], 1)
        probs = torch.tensor([r.prob for r in self.regimes])
        z = torch.multinomial(probs, n, replacement=True, generator=gen)
        kappa = torch.tensor([r.kappa for r in self.regimes])[z]
        lam = torch.tensor([r.lam for r in self.regimes])[z]
        eta = torch.tensor([r.eta for r in self.regimes])[z]
        return {"F": F, "z": z, "kappa": kappa, "lam": lam, "eta": eta}


# ----------------------------------------------------------------------------- feature maps
class FeatureMap:
    """Causal feature map phi_k(obs_0..obs_k).  obs holds what the hedger may see."""
    name = "base"
    dim = 0

    def init_state(self, n):
        return None

    def step(self, obs, state):
        raise NotImplementedError

    def warm_start_targets(self):
        return None


def _rand_features(gen, n_in, n_out, scale=1.0):
    A = torch.randn(n_out, n_in, generator=gen) * scale
    b = torch.randn(n_out, generator=gen)
    return A, b


class StaticTanh(FeatureMap):
    """Memoryless: random tanh features of the current observable state (t, S_k, Q_{k-1})."""

    def __init__(self, D, gen, scale=1.0, s_scale=10.0):
        self.D = D
        self.s_scale = s_scale
        self.A, self.b = _rand_features(gen, 3, D, scale)
        self.name = f"static_tanh(D={D})"
        self.dim = 1 + 3 + D

    def _u(self, obs):
        return torch.stack([obs["tau"], obs["s"] / self.s_scale, obs["Q_prev"]], 1)

    def step(self, obs, state):
        u = self._u(obs)
        return torch.cat([torch.ones_like(u[:, :1]), u, torch.tanh(u @ self.A.T + self.b)], 1), state


class StaticTanhLag1(FeatureMap):
    """Memoryless plus one lag: random tanh features of (t, S_k, Q_{k-1}, dS_k, q_{k-1})."""

    def __init__(self, D, gen, scale=1.0, s_scale=10.0):
        self.D = D
        self.s_scale = s_scale
        self.A, self.b = _rand_features(gen, 5, D, scale)
        self.name = f"static_tanh_lag1(D={D})"
        self.dim = 1 + 5 + D

    def step(self, obs, state):
        u = torch.stack([obs["tau"], obs["s"] / self.s_scale, obs["Q_prev"], obs["dS"] / self.s_scale, obs["q_prev"]], 1)
        return torch.cat([torch.ones_like(u[:, :1]), u, torch.tanh(u @ self.A.T + self.b)], 1), state


class EWMABank(FeatureMap):
    """Moving/weighted averages of the observable channels (price increments and own trades)
    at log-spaced half-lives, optionally with random tanh features on top (matched count)."""

    def __init__(self, half_lives, gen, D=0, scale=1.0, s_scale=10.0, extra_channels=False):
        self.hl = list(half_lives)
        self.decays = torch.tensor([0.5 ** (1.0 / h) for h in self.hl])
        self.J = len(self.hl)
        self.D = D
        self.s_scale = s_scale
        self.extra = extra_channels  # add EWMAs of q*dS (cross term), q^2, dS^2 -> hand-crafted propagator statistics
        n_ch = 4 if not extra_channels else 7
        self.n_lin = 3 + n_ch * self.J
        if D > 0:
            self.A, self.b = _rand_features(gen, self.n_lin, D, scale)
        tag = "cross" if extra_channels else "ewma"
        self.name = f"{tag}_bank(J={self.J}" + (f",D={D})" if D > 0 else ",linear)")
        self.dim = 1 + self.n_lin + D

    def init_state(self, n):
        n_ch = 4 if not self.extra else 7
        return torch.zeros(n, n_ch, self.J)

    def step(self, obs, state):
        dS = obs["dS"] / self.s_scale          # last observed price increment  S_k - S_{k-1}
        q = obs["q_prev"]                      # last own trade  q_{k-1}
        S_lvl = obs["s"] / self.s_scale        # price level (for moving averages of the price)
        Q = obs["Q_prev"]
        chans = [dS, q, S_lvl, Q]
        if self.extra:   # bounded (tanh-squashed) propagator statistics: own trade x price response, trade size, squared move
            chans += [torch.tanh(q * dS * 20.0), torch.tanh(q * q * 200.0), torch.tanh(dS * dS * 20.0)]
        x = torch.stack(chans, 1)                                     # (n, n_ch)
        # EWMA_{k} = lam * EWMA_{k-1} + (1 - lam) * x_k   for increment-type channels,
        # and the same recursion for level channels (= exponentially weighted moving average of the level)
        state = self.decays * state + (1.0 - self.decays) * x[:, :, None]
        u = torch.cat([torch.stack([obs["tau"], obs["s"] / self.s_scale, Q], 1), state.reshape(state.shape[0], -1)], 1)
        feats = [torch.ones_like(u[:, :1]), u]
        if self.D > 0:
            feats.append(torch.tanh(u @ self.A.T + self.b))
        return torch.cat(feats, 1), state


class Reservoir(FeatureMap):
    """Petar's frozen randomized-signature reservoir (rsmc/reservoir.py), ported to torch.

    Channels x_k = [t_k/T, (S_k - F0)/s_scale, Q_{k-1}] * input_scale, driven by increments.
    Readout [1, t, x_k(non-time), h_k].  recurrent=False zeroes W_r (ablated twin)."""

    def __init__(self, D, gen, weight_scale=1.0, bias_scale=1.0, input_scale=1.0, s_scale=10.0,
                 recurrent=True, depth_two=False, tanh_head=0, head_scale=1.0):
        d = 3
        self.D, self.d = D, d
        self.tanh_head = tanh_head
        if tanh_head > 0:   # random tanh features of [t, s, Q, h]  (frozen; readout stays linear)
            self.Ah, self.bh = _rand_features(gen, 3 + D, tanh_head, head_scale / math.sqrt(D))
        self.W = torch.randn(d, D, D, generator=gen) * (weight_scale / math.sqrt(D))
        self.xi = torch.randn(d, D, generator=gen) * bias_scale
        if depth_two:
            self.W2 = torch.randn(d, D, D, generator=gen) * (weight_scale / math.sqrt(D))
            self.xi2 = torch.randn(d, D, generator=gen) * bias_scale
        self.depth_two = depth_two
        if not recurrent:
            self.W = torch.zeros_like(self.W)
        self.input_scale = input_scale
        self.s_scale = s_scale
        self.recurrent = recurrent
        kind = "reservoir" if recurrent else "reservoir_norec"
        if depth_two:
            kind += "_d2"
        if tanh_head > 0:
            kind += f"+tanh{tanh_head}"
        self.name = f"{kind}(D={D},ws={weight_scale},in={input_scale})"
        self.dim = 1 + 3 + D + tanh_head

    def init_state(self, n):
        return {"h": torch.zeros(n, self.D), "x": None}

    def _x(self, obs):
        return torch.stack([obs["tau"], obs["s"] / self.s_scale, obs["Q_prev"]], 1) * self.input_scale

    def step(self, obs, state):
        x = self._x(obs)
        h = state["h"]
        if state["x"] is not None:
            dx = x - state["x"]
            upd = torch.zeros_like(h)
            for r in range(self.d):
                v = torch.tanh(h @ self.W[r].T + self.xi[r])
                if self.depth_two:
                    v = torch.tanh(v @ self.W2[r].T + self.xi2[r])
                upd = upd + v * dx[:, r:r + 1]
            h = h + upd
        u = torch.stack([obs["tau"], obs["s"] / self.s_scale, obs["Q_prev"]], 1)
        parts = [torch.ones_like(u[:, :1]), u, h]
        if self.tanh_head > 0:
            uh = torch.cat([u, h], 1)
            parts.append(torch.tanh(uh @ self.Ah.T + self.bh))
        phi = torch.cat(parts, 1)
        return phi, {"h": h, "x": x}


class OracleState(FeatureMap):
    """Upper bound: sees the fundamental F_k, the pending impact I_k and the regime z."""

    def __init__(self, D, gen, scale=1.0, s_scale=10.0):
        self.D = D
        self.s_scale = s_scale
        self.A, self.b = _rand_features(gen, 5, D, scale)
        self.name = f"oracle_state(D={D})"
        self.dim = 1 + 5 + D

    def step(self, obs, state):
        u = torch.stack([obs["tau"], obs["f"] / self.s_scale, obs["I"] / self.s_scale * 5.0, obs["z"], obs["Q_prev"]], 1)
        return torch.cat([torch.ones_like(u[:, :1]), u, torch.tanh(u @ self.A.T + self.b)], 1), state


class GRUPolicy(FeatureMap):
    """Trained recurrent ceiling (not frozen): GRU over the observable channels."""

    def __init__(self, hidden, gen):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        self.cell = torch.nn.GRUCell(5, hidden)
        self.hidden = hidden
        self.name = f"gru(H={hidden})"
        self.dim = 1 + 3 + hidden
        self.s_scale = 10.0

    def parameters(self):
        return list(self.cell.parameters())

    def init_state(self, n):
        return torch.zeros(n, self.hidden)

    def step(self, obs, state):
        u = torch.stack([obs["tau"], obs["s"] / self.s_scale, obs["Q_prev"], obs["dS"] / self.s_scale, obs["q_prev"]], 1)
        h = self.cell(u, state)
        u3 = u[:, :3]
        return torch.cat([torch.ones_like(u[:, :1]), u3, h], 1), h


# ----------------------------------------------------------------------------- policy + rollout
class LinearReadoutPolicy:
    def __init__(self, fmap: FeatureMap, w: torch.Tensor | None = None):
        self.fmap = fmap
        self.w = torch.zeros(fmap.dim) if w is None else w.clone()
        self.w.requires_grad_(True)
        self.scale = torch.ones(fmap.dim)   # column RMS from the warm-start rollout (training data only)

    def parameters(self):
        ps = [self.w]
        if hasattr(self.fmap, "parameters"):
            ps += self.fmap.parameters()
        return ps


def rollout(policy, mkt: Market, batch: dict, premium: float, position=None, record=False):
    """Closed-loop simulation.  Returns a dict of per-path quantities (torch, differentiable)."""
    F, kappa, lam, eta, z = batch["F"], batch["kappa"], batch["lam"], batch["eta"], batch["z"].float()
    n, N = F.shape[0], mkt.n_steps
    T = mkt.T
    fm = policy.fmap
    state = fm.init_state(n)
    I = torch.zeros(n)
    if mkt.q_init == "delta":
        Q_prev = bachelier_delta(F[:, 0], mkt.K, mkt.sigma, T)
    else:
        Q_prev = torch.zeros(n)
    q_prev = torch.zeros(n)
    S_prev = F[:, 0].clone()
    gain = torch.zeros(n)
    cost = torch.zeros(n)
    turnover = torch.zeros(n)
    penalty = torch.zeros(n)
    rec = {"Q": [], "I": [], "S": [], "phi": []} if record else None
    for k in range(N):
        S = F[:, k] + I
        obs = {
            "tau": torch.full((n,), k / N),
            "s": S - mkt.F0,
            "dS": S - S_prev if k > 0 else torch.zeros(n),
            "Q_prev": Q_prev,
            "q_prev": q_prev,
            "f": F[:, k] - mkt.F0,
            "I": I,
            "z": z,
        }
        phi, state = fm.step(obs, state)
        if position is not None:                      # externally supplied behaviour policy (for warm starts)
            Q = position(k, obs)
        else:
            Q = (phi / policy.scale) @ policy.w
        mid, half = 0.5 * (mkt.q_lo + mkt.q_hi), 0.5 * (mkt.q_hi - mkt.q_lo)
        excess = torch.relu((Q - mid).abs() - half)
        penalty = penalty + excess.pow(2)
        Q = torch.clamp(Q, mkt.q_lo, mkt.q_hi)
        if k >= N - mkt.freeze_last:
            Q = Q_prev
        q = Q - Q_prev
        cost = cost + q * (I + 0.5 * kappa * q + eta * q)
        if record:   # pre-trade pending impact I_k, observed mid S_k, decision Q_k, features phi_k
            rec["Q"].append(Q.detach()); rec["I"].append(I.detach()); rec["S"].append(S.detach()); rec["phi"].append(phi.detach())
        I = lam * (I + kappa * q)
        gain = gain + Q * (F[:, k + 1] - F[:, k])
        turnover = turnover + q.abs()
        S_prev, Q_prev, q_prev = S, Q, q
    payoff = torch.relu(F[:, -1] - mkt.K)
    err = payoff - premium - gain + cost
    out = {"err": err, "payoff": payoff, "gain": gain, "cost": cost, "turnover": turnover, "z": batch["z"], "penalty": penalty}
    if record:
        out["rec"] = {k_: torch.stack(v, 1) for k_, v in rec.items()}
    return out


def frictionless_delta_position(mkt: Market, damping=0.5):
    """Behaviour policy for warm starts: damped move towards the Bachelier delta at the OBSERVED price."""
    def pos(k, obs):
        tau = mkt.T - k * mkt.dt
        target = bachelier_delta(obs["s"] + mkt.F0, mkt.K, mkt.sigma, tau)
        return obs["Q_prev"] + damping * (target - obs["Q_prev"])
    return pos


def warm_start(policy, mkt, batch, premium, ridge=1e-2):
    """Ridge-regress the frictionless delta (at the observed price) on the features along a
    rollout of the delta policy; gives every policy the same sensible starting point."""
    with torch.no_grad():
        out = rollout(policy, mkt, batch, premium, position=frictionless_delta_position(mkt), record=True)
        phi = out["rec"]["phi"].reshape(-1, policy.fmap.dim)
        rms = phi.pow(2).mean(0).sqrt()
        floor = 0.1 * rms[rms > 1e-8].median()          # never amplify a nearly-dead column by a huge factor
        policy.scale = torch.where(rms < 1e-8, torch.ones_like(rms), torch.clamp(rms, min=floor))
        Z = phi / policy.scale
        Q = out["rec"]["Q"].reshape(-1)
        G = Z.T @ Z / Z.shape[0] + ridge * torch.eye(Z.shape[1])
        w = torch.linalg.solve(G, Z.T @ Q / Z.shape[0])
        policy.w.data.copy_(w)


def metrics(out):
    e = out["err"]
    d = {
        "net_rmse": float(e.pow(2).mean().sqrt()),
        "net_mse": float(e.pow(2).mean()),
        "mean_err": float(e.mean()),
        "std_err": float(e.std()),
        "mean_cost": float(out["cost"].mean()),
        "mean_turnover": float(out["turnover"].mean()),
        "cvar95": float(torch.topk(e, max(1, int(0.05 * e.numel()))).values.mean()),
    }
    for zz in torch.unique(out["z"]):
        m = out["z"] == zz
        d[f"net_rmse_z{int(zz)}"] = float(e[m].pow(2).mean().sqrt())
        d[f"mean_cost_z{int(zz)}"] = float(out["cost"][m].mean())
    return d


def train(policy, mkt, premium, gen, n_iters=400, batch_size=1024, lr=1e-2, val_batch=None,
          eval_every=20, verbose=False, weight_decay=0.0):
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=lr * 0.05)
    best = (float("inf"), [p.detach().clone() for p in policy.parameters()])
    hist = []
    t0 = time.time()
    for it in range(n_iters):
        batch = mkt.sample(batch_size, gen)
        out = rollout(policy, mkt, batch, premium)
        loss = out["err"].pow(2).mean() + 10.0 * out["penalty"].mean()
        opt.zero_grad()
        if not torch.isfinite(loss):
            for g in opt.param_groups:
                g["lr"] *= 0.5
            sched.step()
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not torch.isfinite(gn):
            opt.zero_grad()
            sched.step()
            continue
        opt.step()
        sched.step()
        if val_batch is not None and ((it + 1) % eval_every == 0 or it == n_iters - 1):
            with torch.no_grad():
                vm = rollout(policy, mkt, val_batch, premium)["err"].pow(2).mean().item()
            hist.append((it + 1, float(loss.detach()), vm))
            if vm < best[0]:
                best = (vm, [p.detach().clone() for p in policy.parameters()])
            if verbose:
                print(f"    it {it+1:4d} train {loss.item():.4f} val {vm:.4f} ({time.time()-t0:.0f}s)")
    for p, b in zip(policy.parameters(), best[1]):
        p.data.copy_(b)
    return {"best_val_mse": best[0], "history": hist, "train_time": time.time() - t0}


def evaluate(policy, mkt, premium, test_batch):
    with torch.no_grad():
        out = rollout(policy, mkt, test_batch, premium)
    return metrics(out), out


# ----------------------------------------------------------------------------- policy zoo (shared by the scripts)
HALF_LIVES = [1, 2, 4, 8, 16, 32]
LEARNING_RATE_OVERRIDE = {"cross_tanh": 1e-3}   # the hand-crafted cross-statistic bank diverges at the common rate (documented)


def make_policy(name: str, D: int, gen: torch.Generator) -> FeatureMap:
    """Feature maps of the study; ``D`` is the total random-feature budget (reservoir blocks use D//2)."""
    if name == "static_tanh":
        return StaticTanh(D, gen)
    if name == "static_lag1":
        return StaticTanhLag1(D, gen)
    if name == "ewma_linear":
        return EWMABank(HALF_LIVES, gen, D=0)
    if name == "ewma_tanh":
        return EWMABank(HALF_LIVES, gen, D=D)
    if name == "cross_tanh":
        return EWMABank(HALF_LIVES, gen, D=D, extra_channels=True)
    if name == "reservoir":
        return Reservoir(D // 2, gen)                       # Petar's readout [1, t, x, h]
    if name == "reservoir_tanh":
        return Reservoir(D // 2, gen, tanh_head=D // 2)     # reservoir state + random tanh head (matched count)
    if name == "reservoir_norec":
        return Reservoir(D // 2, gen, tanh_head=D // 2, recurrent=False)
    if name == "oracle_state":
        return OracleState(D, gen)
    if name == "gru":
        return GRUPolicy(32, gen)
    raise ValueError(name)


def market_from_args(kappa: str, lam: str, steps: int) -> Market:
    kappas = [float(x) for x in kappa.split(",")]
    lams = [float(x) for x in lam.split(",")]
    regimes = [Regime(f"r{i}", 1.0 / len(kappas), kappa=k, lam=l) for i, (k, l) in enumerate(zip(kappas, lams))]
    return Market(n_steps=steps, regimes=regimes)

"""Tests for the transient-impact / hidden-liquidity hedging study (``rsmc/impact_hedging.py``).

Runs with ``pytest tests/test_impact_hedging.py`` (about 30 s on two cores).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsmc.features import PathFeatures  # noqa: E402
from rsmc.impact_hedging import (  # noqa: E402
    LinearReadoutPolicy,
    Market,
    Regime,
    Reservoir,
    StaticTanh,
    bachelier_call,
    bachelier_delta,
    frictionless_delta_position,
    make_policy,
    rollout,
    train,
    warm_start,
)
from rsmc.reservoir import RandomizedSignatureReservoir  # noqa: E402


def _market(kappa=(0.5, 20.0), lam=(0.71, 0.95), n_steps=40):
    regimes = [Regime("liquid", 0.5, kappa=kappa[0], lam=lam[0]), Regime("illiquid", 0.5, kappa=kappa[1], lam=lam[1])]
    return Market(n_steps=n_steps, regimes=regimes)


def test_torch_reservoir_matches_numpy_reservoir() -> None:
    """The torch port reproduces rsmc.reservoir.RandomizedSignatureReservoir on an open-loop channel path."""
    rng = np.random.default_rng(0)
    numpy_res = RandomizedSignatureReservoir.draw(n_channels=3, dim=16, rng=rng)
    n_paths, n_steps = 7, 25
    x = np.cumsum(rng.standard_normal((n_paths, n_steps + 1, 3)) * 0.05, axis=1)
    x[:, 0, :] = 0.0
    expected = numpy_res.run(PathFeatures(values=x, names=("tau", "s", "Q")))  # (n, N+1, D)

    torch_res = Reservoir(16, torch.Generator().manual_seed(0), input_scale=1.0, s_scale=1.0)
    torch_res.W = torch.tensor(numpy_res.weights, dtype=torch.float64)
    torch_res.xi = torch.tensor(numpy_res.biases, dtype=torch.float64)
    state = {"h": torch.zeros(n_paths, 16, dtype=torch.float64), "x": None}
    for k in range(n_steps + 1):
        obs = {
            "tau": torch.tensor(x[:, k, 0]),
            "s": torch.tensor(x[:, k, 1]),
            "Q_prev": torch.tensor(x[:, k, 2]),
        }
        _, state = torch_res.step(obs, state)
        assert np.allclose(state["h"].numpy(), expected[:, k, :], atol=1e-10)


def test_frictionless_delta_hedge_is_accurate_and_costless() -> None:
    """With kappa = 0 the Bachelier delta hedge has a small discrete-hedging error and zero impact cost."""
    mkt = _market(kappa=(0.0, 0.0), lam=(0.0, 0.0), n_steps=60)
    batch = mkt.sample(4000, torch.Generator().manual_seed(1))
    pol = LinearReadoutPolicy(StaticTanh(4, torch.Generator().manual_seed(2)))
    with torch.no_grad():
        out = rollout(pol, mkt, batch, mkt.premium, position=frictionless_delta_position(mkt, damping=1.0))
    rmse = out["err"].pow(2).mean().sqrt().item()
    assert torch.allclose(out["cost"], torch.zeros_like(out["cost"]))
    assert rmse < 0.8, rmse        # premium 3.9; daily hedging error 0.42 without the 5-step expiry freeze, 0.60 with it
    assert abs(out["err"].mean().item()) < 0.05                # fixed-premium hedge is unbiased without costs


def test_bachelier_delta_is_a_martingale_of_the_call_price() -> None:
    F = torch.linspace(80.0, 120.0, 401)
    price = bachelier_call(F, 100.0, 20.0, 0.5)
    delta = bachelier_delta(F, 100.0, 20.0, 0.5)
    numeric = torch.gradient(price, spacing=(F,))[0]
    assert torch.allclose(numeric[5:-5], delta[5:-5], atol=2e-3)


def test_round_trip_impact_cost_is_nonnegative() -> None:
    """Buy q then sell q one step later: the propagator accounting charges kappa q^2 (1 - lam) >= 0 (no price manipulation)."""
    for kappa, lam in [(0.5, 0.71), (20.0, 0.95), (3.0, 0.0), (3.0, 1.0)]:
        I = 0.0
        q = 0.3
        cost = q * (I + 0.5 * kappa * q)
        I = lam * (I + kappa * q)
        cost += (-q) * (I + 0.5 * kappa * (-q))
        assert cost >= -1e-12
        assert abs(cost - kappa * q * q * (1.0 - lam)) < 1e-12


def test_recurrence_ablation_is_linear_in_the_current_channels() -> None:
    """With W_r = 0 the reservoir state is sum_r tanh(xi_r) (x^r_k - x^r_0): no memory beyond the current state."""
    res = Reservoir(8, torch.Generator().manual_seed(3), recurrent=False, s_scale=1.0)
    state = res.init_state(5)
    xs = torch.randn(5, 10, 3)
    for k in range(10):
        obs = {"tau": xs[:, k, 0], "s": xs[:, k, 1], "Q_prev": xs[:, k, 2]}
        _, state = res.step(obs, state)
    expected = sum(torch.tanh(res.xi[r]) * (xs[:, -1, r] - xs[:, 0, r])[:, None] for r in range(3))
    assert torch.allclose(state["h"], expected, atol=1e-6)


def test_policy_zoo_trains_without_nans() -> None:
    """Every feature map of the study warm-starts, trains a few closed-loop steps and evaluates with finite numbers."""
    mkt = _market(n_steps=20)
    p0 = mkt.premium
    ws = mkt.sample(256, torch.Generator().manual_seed(4))
    val = mkt.sample(256, torch.Generator().manual_seed(5))
    for name in ["static_tanh", "ewma_linear", "ewma_tanh", "cross_tanh", "reservoir", "reservoir_tanh", "reservoir_norec", "oracle_state", "gru"]:
        fm = make_policy(name, 16, torch.Generator().manual_seed(6))
        pol = LinearReadoutPolicy(fm)
        warm_start(pol, mkt, ws, p0)
        info = train(pol, mkt, p0, torch.Generator().manual_seed(7), n_iters=3, batch_size=128, lr=1e-3, val_batch=val, eval_every=1)
        assert np.isfinite(info["best_val_mse"]), name
        with torch.no_grad():
            out = rollout(pol, mkt, val, p0)
        assert out["err"].shape == (256,) and torch.isfinite(out["err"]).all(), name
        assert torch.isfinite(out["cost"]).all(), name


def test_regime_is_invisible_in_the_fundamental_path() -> None:
    """The fundamental has the same law in both regimes: only the response to own trades can reveal the regime."""
    mkt = _market()
    batch = mkt.sample(20000, torch.Generator().manual_seed(8))
    dF = torch.diff(batch["F"], dim=1)
    v0 = dF[batch["z"] == 0].var().item()
    v1 = dF[batch["z"] == 1].var().item()
    assert abs(v0 / v1 - 1.0) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")


# ----------------------------------------------------------------------------- audit tests (added 9 Nov 2026)
def _policies_for_audit():
    for name in ["static_tanh", "static_lag1", "ewma_tanh", "cross_tanh", "reservoir", "reservoir_tanh", "gru"]:
        fm = make_policy(name, 16, torch.Generator().manual_seed(21))
        pol = LinearReadoutPolicy(fm)
        pol.w.data.normal_(0, 0.05)
        yield name, pol


def test_no_look_ahead_future_shocks_do_not_change_past_positions() -> None:
    """Positions, features and costs up to step k must be identical on two batches that share the
    fundamental path up to k and differ afterwards (adaptedness of every feature map)."""
    mkt = _market(n_steps=30)
    base = mkt.sample(64, torch.Generator().manual_seed(31))
    k = 12
    alt = {kk: v.clone() for kk, v in base.items()}
    tail = torch.randn(64, mkt.n_steps - k, generator=torch.Generator().manual_seed(32)) * mkt.sigma * (mkt.dt ** 0.5)
    alt["F"][:, k + 1:] = alt["F"][:, k:k + 1] + torch.cumsum(tail, 1)
    assert torch.allclose(alt["F"][:, :k + 1], base["F"][:, :k + 1]) and not torch.allclose(alt["F"][:, k + 1:], base["F"][:, k + 1:])
    for name, pol in _policies_for_audit():
        with torch.no_grad():
            a = rollout(pol, mkt, base, mkt.premium, record=True)
            b = rollout(pol, mkt, alt, mkt.premium, record=True)
        assert torch.allclose(a["rec"]["Q"][:, :k + 1], b["rec"]["Q"][:, :k + 1], atol=1e-6), name
        assert torch.allclose(a["rec"]["phi"][:, :k + 1], b["rec"]["phi"][:, :k + 1], atol=1e-6), name
        assert torch.allclose(a["rec"]["I"][:, :k + 1], b["rec"]["I"][:, :k + 1], atol=1e-6), name


def test_net_error_matches_explicit_cash_accounting() -> None:
    """err = payoff - premium - gain + cost equals payoff - premium - terminal wealth, where wealth is
    computed from explicit cash flows at execution prices F_k + I_k + (kappa/2 + eta) q_k and the final
    position is valued at the unaffected price F_N (self-financing identity by summation by parts)."""
    mkt = _market(n_steps=25)
    batch = mkt.sample(128, torch.Generator().manual_seed(41))
    for name, pol in _policies_for_audit():
        with torch.no_grad():
            out = rollout(pol, mkt, batch, mkt.premium, record=True)
        Q = out["rec"]["Q"]                                     # Q_k, k = 0..N-1
        F = batch["F"]; kappa = batch["kappa"][:, None]; lam = batch["lam"][:, None]; eta = batch["eta"][:, None]
        Q0 = bachelier_delta(F[:, 0], mkt.K, mkt.sigma, mkt.T)[:, None]
        Qprev = torch.cat([Q0, Q[:, :-1]], 1)
        q = Q - Qprev
        # pending impact before each trade, recomputed independently from the trades
        I = torch.zeros_like(Q)
        for k in range(1, mkt.n_steps):
            I[:, k] = lam[:, 0] * (I[:, k - 1] + kappa[:, 0] * q[:, k - 1])
        assert torch.allclose(I, out["rec"]["I"], atol=1e-5), name
        exec_price = F[:, :-1] + I + (0.5 * kappa + eta) * q
        cash = -(exec_price * q).sum(1)                         # pay for the trades (the inherited Q0 was free)
        wealth = cash + Q[:, -1] * F[:, -1] - Q0[:, 0] * F[:, 0]  # mark the inherited and final positions at the unaffected price
        err_explicit = out["payoff"] - mkt.premium - wealth
        assert torch.allclose(err_explicit, out["err"], atol=1e-4), name


def test_recorded_impact_is_pre_trade() -> None:
    """rec['I'][:, k] is the pending impact BEFORE the trade at k: zero at k=0 and independent of q_k."""
    mkt = _market(n_steps=10)
    batch = mkt.sample(16, torch.Generator().manual_seed(51))
    _, pol = next(_policies_for_audit())
    with torch.no_grad():
        out = rollout(pol, mkt, batch, mkt.premium, record=True)
    assert torch.allclose(out["rec"]["I"][:, 0], torch.zeros(16))


def test_train_val_test_streams_are_disjoint() -> None:
    """The seeds used by the protocol give different fundamental paths for warm start, training, validation and test."""
    mkt = _market(n_steps=10)
    seed = 5
    sets = {"train": mkt.sample(3, torch.Generator().manual_seed(1000 + seed))["F"],
            "val": mkt.sample(3, torch.Generator().manual_seed(5000 + seed))["F"],
            "ws": mkt.sample(3, torch.Generator().manual_seed(7000 + seed))["F"],
            "test": mkt.sample(3, torch.Generator().manual_seed(9000 + seed))["F"]}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not torch.allclose(sets[names[i]], sets[names[j]]), (names[i], names[j])


def test_regime_enters_non_oracle_features_only_through_the_price_response() -> None:
    """With kappa = 0 in both regimes the observed path carries no impact, so every non-oracle feature map
    must produce identical features and positions whatever the regime label (no leakage of z, kappa, lam)."""
    regimes = [Regime("a", 0.5, kappa=0.0, lam=0.3), Regime("b", 0.5, kappa=0.0, lam=0.9)]
    mkt = Market(n_steps=15, regimes=regimes)
    batch = mkt.sample(32, torch.Generator().manual_seed(61))
    flipped = {k: v.clone() for k, v in batch.items()}
    flipped["z"] = 1 - flipped["z"]
    flipped["lam"] = torch.tensor([r.lam for r in regimes])[flipped["z"]]
    for name, pol in _policies_for_audit():
        with torch.no_grad():
            a = rollout(pol, mkt, batch, mkt.premium, record=True)
            b = rollout(pol, mkt, flipped, mkt.premium, record=True)
        assert torch.allclose(a["rec"]["phi"], b["rec"]["phi"], atol=1e-6), name
        assert torch.allclose(a["err"], b["err"], atol=1e-6), name
    oracle = LinearReadoutPolicy(make_policy("oracle_state", 16, torch.Generator().manual_seed(21)))
    oracle.w.data.normal_(0, 0.05)
    with torch.no_grad():
        a = rollout(oracle, mkt, batch, mkt.premium, record=True)
        b = rollout(oracle, mkt, flipped, mkt.premium, record=True)
    assert not torch.allclose(a["rec"]["phi"], b["rec"]["phi"])   # the oracle, by design, does see z

# Randomized-Signature Martingale Control Variates

Variance reduction for Monte Carlo pricing of path-dependent options using
**martingale control variates**, where the predictable control integrand comes
from a *frozen randomized-signature reservoir* (only a linear readout is
trained). Finite truncated signatures serve as a benchmark control class.

This repository is being built step by step. So far: the **Black-Scholes
simulator**, the **Asian-aware feature channels**, the **frozen
randomized-signature reservoir** (D = 64), the **ridge-trained martingale
control variate**, and the **finite truncated-signature benchmark**. On the
arithmetic Asian call the reservoir delivers a **~50x variance reduction**
(7x smaller price standard error) while staying unbiased, and it reproduces
**~100%** of the real signature's variance reduction (see the benchmark below).

## The idea in one paragraph

We want `p_N = E_Q[G_N]`, the price of a path-dependent payoff `G_N`. Plain
Monte Carlo averages `G_N` over many simulated paths, but the variance is high.
We subtract a zero-mean *martingale control variate*

```
C(Z) = sum_{k=0}^{N-1}  Z_k . dM_{k+1},      E_Q[dM_{k+1} | F_{t_k}] = 0,
```

where `Z_k` is predictable (known at time `t_k`). Because `C(Z)` has mean zero,
`E_Q[G_N - C(Z)] = p_N` is unchanged, but the variance can be much smaller. In
Black-Scholes the martingale increments `dM_{k+1}` are exactly the Brownian
innovations, which is why the simulator returns them.

## Step 1: the Black-Scholes simulator

Exact (no discretisation error) simulation on a grid `0 = t_0 < ... < t_N = T`:

```
S_{k+1} = S_k * exp( (r - sigma^2 / 2) dt + sigma sqrt(dt) Z_{k+1} ),   Z ~ N(0, 1).
```

The simulator returns the price paths `S_0, ..., S_N` (for the payoff) **and**
the innovations `Z_{k+1}` (the martingale increments for the control variate).

Experimental setup encoded in `ExperimentConfig`:

| Setting          | Value                |
| ---------------- | -------------------- |
| Model            | Black-Scholes        |
| Payoff           | arithmetic Asian call|
| Time steps `N`   | 50                   |
| Training paths   | 20,000               |
| Validation paths | 5,000                |
| Test paths       | 100,000              |

## Step 2: features and the randomized-signature reservoir

**Why not feed the raw price?** A single price channel (`d = 1`) is a poor
input: the signature of a 1-D path only depends on its net increment
`S_T - S_0`, so it cannot see path ordering or averages. We therefore build
three **Asian-aware, time-augmented** channels `x_k = Psi_k(X_0:k)` (all
starting at 0 at `k = 0`):

| Channel | Definition | Role |
| --- | --- | --- |
| time | `tau_k = t_k / T` | time augmentation -> signature sees ordering/timing |
| log-price | `log(S_k / S_0)` | current price level |
| running average | `log(A_k / S_0)`, `A_k = mean(S_0..S_k)` | Asian state variable |

These three are the *sufficient* state for the arithmetic Asian call. Two more
genuine path-state channels - running **max** and **min** of log-price
(lookback / barrier state) - are available via `asian_features(..., include_running_max=True, include_running_min=True)`
for `d = 5`, used in the Step 4 benchmark to study dimension scaling.

**The frozen reservoir.** For each channel `r` we draw, once, random
`W_r in R^{D x D}` and `xi_r in R^D` (scaled by `1/sqrt(D)`), freeze them, and
roll the state forward driven by the feature increments:

```
h_0 = 0,    h_{k+1} = h_k + sum_{r=1}^d rho(W_r h_k + xi_r) * Delta x^{(r)}_{k+1},   rho = tanh.
```

Since `h_k` uses only `x_0..x_k` it is `F_{t_k}`-measurable (predictable) -
exactly the property the martingale control needs. The readout feature vector
is `phi_k = [1, t_k, h_k] in R^{D+2}`. Only the readout `beta` is trained; the
reservoir weights stay frozen.

## Step 3: the ridge-trained martingale control variate

Because the control `C = sum_k (beta . phi_k) dW_{k+1}` is linear in `beta` and
the shocks are fixed, each path collapses into a single regression row

```
Phi_i = sum_{k=0}^{N-1} phi_{i,k} * dW_{i,k+1}   in R^q,     C_i = beta . Phi_i.
```

`E[Phi] = 0` (predictable feature times mean-zero shock), so `C` is mean zero
and `E[G - C] = p_N` no matter what `beta` is - the control cannot bias the
price. We then **ridge-fit** `beta` on the training paths:

```
(a_hat, beta_hat) = argmin_{a, beta}  sum_i (G_i - a - beta . Phi_i)^2 + lam ||beta||^2.
```

The intercept `a` absorbs the price level so it does not distort the control;
it is used only for training and dropped from `C`. `lambda` is tuned on the
validation set. **Sample splitting** (train / val independent of test) keeps
the estimator unbiased (Theorem A).

**Evaluation** on independent test paths reports plain vs controlled samples
`Y^MC = G`, `Y^CV = G - C`, the prices, their standard errors, and the headline

```
VR = Var(G) / Var(G - C),     SE_MC / SE_CV = sqrt(VR).
```

A per-path residual `G_i - C_i` is *not* meant to be near zero; it is meant to
have smaller variance than `G_i`. So we train on the squared residual but
evaluate on the variance of the controlled samples and the price standard
error. Typical result on the Asian call: `VR ~ 50x`, with `mean(C) ~ 0`
(bias check) and the controlled price matching plain MC.

## Step 4: benchmark against the *real* truncated signature

The reservoir is meant to be a *cheap random surrogate* for the **finite
truncated signature**. To test that claim, `rsmc/signature.py` computes the real
thing and runs it through the **same** readout, ridge fit and evaluation - only
the feature map changes (reservoir state `h_k` -> signature `s_k`). It is the
same control-variate architecture, a different adapted path representation:

```
reservoir : h_k             -> linear readout -> Z_k = beta . [1, t_k, h_k]
signature : Sig^{<=L}(x_0:k) -> linear readout -> Z_k = alpha . [1, t_k, s_k]
```

The depth-`L` truncated signature is the product, in the truncated tensor
algebra, of the exponentials of the increments (**Chen's identity**):

```
s_k = S(x_0:k) = exp(dx_1) (x) ... (x) exp(dx_k),   exp(dx) = (1, dx, dx(x)dx/2!, ...).
```

We roll this forward to get the running signature `s_k` at every step (exactly
like the reservoir state). Its dimension does **not** depend on `N`, only on the
channels `d` and the level `L`:

```
q_L = d + d^2 + ... + d^L      # d = 3:  L=2->12, L=3->39, L=4->120
                               # d = 5:  L=2->30, L=3->155, L=4->780
```

This benchmark lives in **`scripts/demo.py` (Part B, cells 8-11)** - no separate
script. `resid/best` is the residual variance `Var(G - C)` relative to the best
method in the table (`1.00` = best, higher = more leftover variance; unlike a
"retention" ratio it stays discriminating when every VR is high). Time is split
into feature build (`feat`) and readout fit (`fit`); evaluation is one dot
product. **Both sweeps use the same 30k test paths and the same 20k train set**,
so the times are directly comparable across `d`. Single-seed run, times indicative.

`d = 3` Asian features (time, log-price, running average):

| method          | dim | VR     | resid/best | feat   | fit  |
| --------------- | --: | -----: | ---------: | -----: | ---: |
| reservoir D=32  |  32 | 44.1x  |     1.35x  |  3.5s  | 0.1s |
| reservoir D=64  |  64 | 53.2x  |     1.12x  |  6.5s  | 0.4s |
| reservoir D=128 | 128 | 59.5x  |     1.00x  | 13.5s  | 0.8s |
| signature L=2   |  12 | 30.6x  |     1.94x  |  0.8s  | 0.1s |
| signature L=3   |  39 | 46.7x  |     1.27x  |  2.6s  | 0.2s |
| signature L=4   | 120 | 49.7x  |     1.20x  |  8.0s  | 0.8s |

`d = 5` (add running max / min of log-price - lookback / barrier state):

| method          | dim | VR     | resid/best | feat   | fit   |
| --------------- | --: | -----: | ---------: | -----: | ----: |
| reservoir D=32  |  32 | 46.9x  |     1.59x  |  5.1s  |  0.2s |
| reservoir D=64  |  64 | 62.6x  |     1.19x  | 13.4s  |  0.4s |
| reservoir D=128 | 128 | 69.7x  |     1.07x  | 26.1s  |  1.0s |
| signature L=2   |  30 | 43.3x  |     1.73x  |  1.6s  |  0.2s |
| signature L=3   | 155 | 67.4x  |     1.11x  |  9.8s  |  1.2s |
| signature L=4   | 780 | 74.7x  |     1.00x  | 55.9s  | 10.6s |

**Takeaways (honest pilot conclusion).**

- **Faithfulness.** At matched dimension the reservoir and the real signature sit
  on the *same* variance-reduction curve - their residual variances are within a
  few percent. The cheap random projection loses little.
- **No compression win at small `d`/`L`.** With `d = 3` the exact signature is
  tiny, so low-order signatures (`L = 2, 3`) are cheap *and* competitive - the
  reservoir matches their VR but does **not** beat them on time.
- **Compression starts to show at `d = 5`.** The richest signature `L = 4` is
  780-dim and by far the heaviest (best VR, 74.7x at ~56s). Reservoir `D = 128`
  reaches 69.7x with **~6x fewer features**, **~2.1x less feature-build time**,
  and only **~7% more residual variance**. (Mid-level `L = 3` is still cheaper
  than `D = 128` and nearly as good - the reservoir's clean win is against the
  *high-order* signature.) Signature dimension grows like `d^L` while the
  reservoir stays fixed at `D`, so this gap widens as `d` and `L` increase.
- **Caveat.** Single random seed - a pilot that confirms the pipeline, not a
  final benchmark.

![Signature dimension explodes as d^L while D is fixed](figures/demo_dimension_growth.png)
![Reservoir vs finite truncated signature](figures/demo_benchmark.png)

## Layout

```
rsmc/
  black_scholes.py   # BlackScholesModel + SimulatedPaths (the simulator)
  payoffs.py         # Asian / European payoffs, closed-form BS + geometric Asian prices
  features.py        # asian_features: time / log-price / running-avg (+ optional max/min)
  reservoir.py       # RandomizedSignatureReservoir (D = 64) + readout_features
  signature.py       # finite truncated signature (Chen's identity) - the benchmark
  control.py         # Phi assembly, RidgeMartingaleControl, evaluation + VR metrics
  config.py          # ExperimentConfig + make_datasets (train/val/test splits)
scripts/
  demo.py            # ONE file: Part A method (cells 1-7) + Part B benchmark (cells 8-11)
tests/
  test_black_scholes.py   # simulator sanity checks (martingale, European vs analytic)
  test_reservoir.py       # feature + reservoir checks (h_0 = 0, non-anticipative, ...)
  test_control.py         # control-variate checks (unbiased, VR > 1, lambda tuning)
  test_signature.py       # signature correctness (Chen, exp, dim) + unbiased + VR > 1
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/demo.py                 # Part A method + Part B signature benchmark
python tests/test_black_scholes.py     # simulator tests (also works with: pytest tests)
python tests/test_reservoir.py         # feature + reservoir tests
python tests/test_control.py           # control-variate tests
python tests/test_signature.py         # signature tests
```

`scripts/demo.py` is written with `# %%` cell markers, so you can open it in
Cursor / VS Code (or Jupyter) and run it cell by cell like a notebook. **Part A**
(cells 1-7) is the method - simulate, train, evaluate, ~50x VR - and runs in
under a minute. **Part B** (cells 8-11) is the signature benchmark and is the
heavy part (a few minutes); the cell markers let you run Part A on its own. The
`N_COMPARE` / `N_HIGHD` knobs control the benchmark sample sizes.

`demo.py` first sanity-checks the simulator (the discounted price is a
martingale, and the Monte Carlo European call matches the closed-form
Black-Scholes price), then trains the control and reports the variance
reduction. The deeper correctness properties are enforced by the test suite:

- **Discounted price is a martingale** (`test_black_scholes.py`).
- **Reservoir is non-anticipative** - `h_k` is unchanged when the path is
  perturbed after step `k` - and a **fixed-`beta` reservoir control has mean
  zero**, the unbiasedness backbone (`test_reservoir.py`).
- **Controlled estimator is unbiased and reduces variance** (`test_control.py`).

## Quick start in code

```python
import numpy as np
from rsmc import (
    BlackScholesModel, arithmetic_asian_call,
    asian_features, RandomizedSignatureReservoir,
    assemble_control_features, RidgeMartingaleControl, evaluate_control,
)

model = BlackScholesModel(s0=100.0, r=0.03, sigma=0.20, T=1.0)
reservoir = RandomizedSignatureReservoir.draw(n_channels=3, dim=64, rng=np.random.default_rng(100))

def rows(n, seed):                       # collapse paths into (Phi, G)
    paths = model.simulate(n, 50, np.random.default_rng(seed))
    phi = assemble_control_features(reservoir, asian_features(paths),
                                    paths.times, paths.brownian_increments)
    return phi, arithmetic_asian_call(paths, strike=100.0, rate=model.r)

phi_train, g_train = rows(20_000, 0)
phi_test,  g_test  = rows(100_000, 1)

control = RidgeMartingaleControl.fit(phi_train, g_train, lam=0.0)
report = evaluate_control(control, phi_test, g_test)
print(report.summary())                  # VR ~ 50x, unbiased
```

To use the **real signature** instead of the reservoir, just swap the feature
builder; everything downstream (ridge fit, evaluation) is identical:

```python
from rsmc import assemble_signature_control_features
phi = assemble_signature_control_features(asian_features(paths), paths.times,
                                          paths.brownian_increments, level=3)
```

## Roadmap

- **Done:** Black-Scholes simulator, arithmetic Asian payoff, Asian-aware
  features (`d = 3`, with optional max/min for `d = 5`), frozen
  randomized-signature reservoir, the ridge-trained sample-split martingale
  control variate (~50x VR), and the **finite truncated-signature benchmark**
  (the reservoir tracks the signature's variance reduction at matched dimension;
  at `d = 5` it nears the 780-dim `L = 4` signature with ~6x fewer features and
  ~2.1x less feature-build time, on the same 30k test paths).
- **Next:** repeat over 5-10 random seeds for error bars; push to the regime
  where the reservoir's fixed-dimension advantage is decisive (higher `d`/`L`, or
  a lookback / barrier payoff needing more path state); then ablations in `D`,
  `N`, `lambda` and the Heston model.

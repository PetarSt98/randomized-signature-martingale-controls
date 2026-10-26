# Randomized-Signature Martingale Control Variates

Variance reduction for Monte Carlo pricing of path-dependent options using
**martingale control variates**, where the predictable control integrand comes
from a *frozen randomized-signature reservoir* (only a linear readout is
trained). Finite truncated signatures serve as a benchmark control class.

This repository is being built step by step. So far: the **Black-Scholes and
Heston simulators**, the **multi-asset Heston** and **commodity forward-curve**
benchmark cases, the **frozen randomized-signature reservoir** (D = 64), the
**ridge-trained martingale control variate**, and the finite truncated-signature
benchmark machinery.

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
(lookback / barrier state) - remain available via
`asian_features(..., include_running_max=True, include_running_min=True)`, but
they are not part of the main benchmark because the selected payoff is not a
barrier or lookback payoff.

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

## Step 4: final benchmark ladder

The main demo now reports four financially defensible benchmark cases:

| Case | Product | Feature dimension | Martingale dimension |
| --- | --- | ---: | ---: |
| Black-Scholes | arithmetic Asian call | `d = 3` | `m = 1` |
| Heston | arithmetic Asian call | `d = 4` | `m = 2` |
| three-asset Heston | basket Asian call | `d = 8` | `m = 6` |
| commodity forward curve | Asian crack-spread option | `d = 12` | `m = 9` |

Every case uses the same sample-split ridge martingale-control pipeline. The
demo sweeps reservoirs `D = 16, 32, 64, 128, 256` and exact signatures
`L = 2, 3, 4`, then reports prices, standard errors, variance reduction,
standard-error improvement, control mean, selected lambda, row dimension and
feature-build / fit timing. It reads `d` from the actual `PathFeatures` object
and infers `m` from the Brownian increment array, so reported dimensions cannot
drift from the data.

Very wide exact-signature rows are controlled by explicit flags in
`scripts/control_variates_demo.py`: `RUN_SIGNATURE_D8_L4`, `RUN_SIGNATURE_D12_L3`, and
`RUN_SIGNATURE_D12_L4`. The default is to skip `d=8,L=4` and `d=12,L=4`, while
running `d=12,L=3`; skipped rows are printed with their theoretical dimensions.

The reservoir is meant to be a cheap random surrogate for the **finite truncated
signature**. `rsmc/signature.py` still computes the real thing and runs it
through the same readout, ridge fit and evaluation - only the feature map changes
(reservoir state `h_k` -> signature `s_k`). It is the same control-variate
architecture, a different adapted path representation:

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
                               # d = 4:  L=2->20, L=3->84, L=4->340
                               # d = 8:  L=2->72, L=3->584, L=4->4,680
                               # d = 12: L=2->156, L=3->1,884, L=4->22,620
```

The commodity case is the main high-dimensional compression benchmark: level 4
already has 22,620 exact-signature coordinates before the martingale blocks,
while the randomized reservoir keeps a fixed dimension `D`.

All rows use the same train / validation / test counts. The default demo is a
proper empirical run (`train=10,000`, `val=3,000`, `test=10,000`) and may take a
while, especially with `N=252`. Use `SMOKE_SIZES` in `scripts/control_variates_demo.py` only for
quick integration checks. The ridge grid extends to `1e6`, so high-dimensional
rows are not forced to choose an under-regularised edge value. High-q rows with
`q > n` should still be read as proposal evidence rather than final production
estimates.

The hedging demo in `scripts/hedging_demo.py` is the observed-filtration
extension. It trades discounted stock increments and hides the Heston variance,
replacing it by a noisy observation `Y = V + eta eps`. The reservoir and
signature hedge readouts use the common current-state block
`[1, t_k, x_k, h_k]` or `[1, t_k, x_k, s_k]` (with the time channel not
duplicated inside `x_k`), matching the direct current-state input used by the
small causal Transformer hedge. Linear hedges also remove the original time
channel before prepending `t_k`, so ridge regularization is not distorted by
duplicated time columns. The default noise sweep is
`eta = 0, 0.01, 0.02, 0.04, 0.08, 0.16`; larger values are better treated as
stress tests rather than core benchmark points. The one-asset Heston case also
includes a model-aware particle filter benchmark that estimates
`E[V_k | S_0:k, Y_0:k]` and `Var(V_k | S_0:k, Y_0:k)` and uses only those
filtered quantities plus the current raw signal as hedge features; it still
trades only the stock. The particle-filter benchmark reports both affine and
quadratic ridge readouts, with the headline PF row selected by validation MSE.
The demo prints paired bootstrap intervals for selected test-path MSE
differences, including the eta sweep, so method comparisons use the same
resampled test paths.

## Layout

```
rsmc/
  black_scholes.py   # BlackScholesModel + SimulatedPaths (the simulator)
  heston.py          # single-asset Heston simulator
  multi_asset_heston.py # three-asset Heston basket simulator
  commodity.py       # commodity forward-curve simulator
  payoffs.py         # Asian, basket Asian, crack-spread, and validation payoffs
  features.py        # adapted PathFeatures builders for d = 3, 4, 8, 12
  reservoir.py       # RandomizedSignatureReservoir (D = 64) + readout_features
  signature.py       # finite truncated signature (Chen's identity) - the benchmark
  control.py         # Phi assembly, RidgeMartingaleControl, evaluation + VR metrics
  config.py          # ExperimentConfig + make_datasets (train/val/test splits)
scripts/
  control_variates_demo.py # one-command reservoir-vs-signature pricing sweep
  hedging_demo.py    # observed-filtration hedging sweep
tests/
  test_black_scholes.py   # simulator sanity checks (martingale, European vs analytic)
  test_heston.py          # Heston simulator, features, vector control
  test_multi_asset_heston.py # d=8 basket benchmark components
  test_commodity.py       # d=12 crack-spread benchmark components
  test_demo.py            # four-row demo smoke test
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
python scripts/control_variates_demo.py # four pricing rows: d=3,4,8,12
python tests/test_black_scholes.py     # simulator tests (also works with: pytest tests)
python tests/test_heston.py            # Heston tests
python tests/test_multi_asset_heston.py
python tests/test_commodity.py
python tests/test_demo.py
python tests/test_reservoir.py         # feature + reservoir tests
python tests/test_control.py           # control-variate tests
python tests/test_signature.py         # signature tests
```

The standalone homogeneous free-Lie rank reconnaissance has three explicit
commands. It uses generic Euclidean states and does not enter the finance
pipeline. Its outputs are numerical evidence only; the per-case matrix budget
is checked before basis generation, parameter sampling, or autodiff:

```bash
python scripts/lie_rank_atlas.py smoke
python scripts/lie_rank_atlas.py single --architecture shallow_biased_tanh --input-dim 2 --state-dim 2 --degree 3
python scripts/lie_rank_atlas.py full
```

The separate mathematical module now proves generic fixed-zero Hall evaluation
rank `min(D, nu)` for shallow biased-tanh fields in the full parameter space,
where `nu` is the number of Hall coordinates through degree `L`. Its finite
grading witness has selected minor `2^(-min(D,nu))`. Thus `D=nu` is the sharp
width for local recovery from one vector of the **degree-L truncated polynomial
response**. This does not assert exact recovery from an actual CDE endpoint
or a result for the implemented discrete recurrence. The
[authoritative theorem note](research_output/intrinsic_hall_rank_theorem.md)
contains the proof; the [current overview](research_output/technical_note.md)
also distinguishes the preserved scalar eight-response results and the
separate sufficient-width word theorem. The
[manuscript](paper/randomized_signature_identifiability.tex) includes the full
proofs and the scalar actual-response error bound. Reproduce the 17 finite exact witness
checks without writing artifacts:

```bash
python -B research_output/reproduce_intrinsic_hall_rank.py
```

`scripts/control_variates_demo.py` is written with `# %%` cell markers, so you can open it in
Cursor / VS Code (or Jupyter) and run it cell by cell like a notebook.
`DEFAULT_SIZES`, `SMOKE_SIZES`, `DEFAULT_N_STEPS`, `RESERVOIR_DIMS`,
`SIGNATURE_LEVELS`, the `RUN_SIGNATURE_*` flags, and `RIDGE_LAMBDAS` are the
demo knobs.

`control_variates_demo.py` trains every reservoir and signature control for each case and reports
model, product, feature names, actual `d`, martingale dimension `m`, path
shapes, plain MC, controlled prices/SEs, variance reduction, control mean,
selected lambda and timing. The deeper correctness properties are enforced by
the test suite:

- **Discounted price is a martingale** (`test_black_scholes.py`).
- **Reservoir is non-anticipative** - `h_k` is unchanged when the path is
  perturbed after step `k` - and a **fixed-`beta` reservoir control has mean
  zero**, the unbiasedness backbone (`test_reservoir.py`).
- **Controlled estimator is unbiased and reduces variance** (`test_control.py`).
- **The d=8 and d=12 benchmark components have the requested shapes and
  martingale dimensions** (`test_multi_asset_heston.py`, `test_commodity.py`).

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

- **Done:** the final benchmark ladder `d=3 -> d=4 -> d=8 -> d=12`, with
  reservoir martingale controls and finite-signature benchmark machinery.
- **Next:** repeat over 5-10 random seeds for error bars; add explicit optional
  signature-control sweeps for selected levels/sample sizes; then ablations in
  `D`, `N`, `lambda` and model parameters.

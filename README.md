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

## CAS: bounded frictionless reservoir comparison

`scripts/cas_frictionless.py` is a separate, fixed-budget experiment. It keeps
the historical pricing, observed-signal and particle-filter demos intact.
The task is the existing one-asset Heston arithmetic Asian call (strike 100,
one year, default model parameters), with 50 monitoring/trading intervals and
4,000 / 2,000 / 8,000 independent training / validation / test paths. The payoff
averages S_1,...,S_N (excluding inception) and is discounted to time zero.
Every method observes only the same three causal channels:
`x_k = [t_k/T, log(S_k/S_0), log(A_k/S_0)]`, where A_k includes S_0,...,S_k.
Neither true variance, synthetic volatility observations nor filters enter.
Trading gains use actual discounted stock increments
`Delta S_tilde_(k+1) = exp(-r t_(k+1)) S_(k+1) - exp(-r t_k) S_k`.

Both reservoirs start at `h_0=0` and use the existing increment recurrence
`h_(k+1) = h_k + sum_i V_i(h_k) Delta x_(k+1)^i`.
The shallow field is `tanh(W_i h + xi_i)`; the sole alternative is
`tanh(B_i tanh(A_i h + b_i) + c_i)`, with inner width equal to state width D.
All matrices are independent Gaussian draws with standard deviation
`weight_scale/sqrt(D)`. Bias standard deviation is `bias_scale=1`, including
both layers of the alternative; biases are not divided by sqrt(D).
Nonzero random biases prevent a dead zero-initialized tanh recurrence.
Weights and biases stay frozen; only linear hedge readouts are trained.
Both readouts are `[1,t_k,x_k_without_time,h_k]`, with no duplicated time.
Depth-two has twice as many frozen parameters at the same D; this comparison
matches state/readout width, not frozen parameter count or computation.

For this dataset partition, the premium p is the discounted training-payoff
mean rounded once to float32 (matching the existing Transformer capital buffer).
That exact value is saved and shared across training, validation, test, all
methods and future cost scenarios using this partition. No method fits an
intercept or recenters evaluation errors. For frozen features, define
`Psi_j = sum_k phi_(j,k) Delta S_tilde_(j,k+1)` and training-only column RMS
`s_l = sqrt(mean_train(Psi_l^2))` (constant-zero columns use scale 1).
The fixed-feature fit is ordinary augmented least squares:

```
Z = Psi_train / s
theta_hat = argmin_theta ||H_train - p - Z theta||^2 + lambda ||theta||^2
beta_hat = theta_hat / s
e_test = H_test - p - Psi_test beta_hat
```

There is no centering of Z, since that would implicitly change the premium.
Lambda multiplies a penalty alongside a **sum** of squared residuals. Selection
uses the **unpenalized mean** squared validation hedging error, averaged over
three reservoir draws. Inputs require only causal deterministic transforms;
the Transformer additionally fits its channel normalization on training data.

The declaration in `results/cas_frictionless/plan.json` lists six complete
`(D, weight_scale, bias_scale, lambda)` tuples per architecture, matched across
architectures. Three validation draws require 36 fits. Each selected tuple is
then fitted on training paths with three fresh reservoir draws and evaluated
on the independent common test paths (six more fits). The simple state hedge
uses the same readout without h and fixed lambda 0.1 (one fit). The existing
Transformer uses its unchanged default architecture/training settings and one
declared seed (one training run; validation selects its epoch). Total: 44 fits,
with no post-result candidate expansion. Unit-test fits are correctness checks,
not members of this experiment. This sparse list does not isolate every tuning
factor; one dataset partition and three draws cannot establish a universal ranking.

Run from the repository root with the prepared environment:

```bash
/workspace/.venvs/rsmc/bin/python scripts/cas_frictionless.py --declare
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/workspace/.cache/rsmc-matplotlib /workspace/.venvs/rsmc/bin/python scripts/cas_frictionless.py
```

The second command also resumes: completed fits are reused; interrupted attempts
are recorded as failures rather than silently repeated beyond the budget.
Results, failures, individual fit/feature/evaluation times, residuals and the
comparison figure are saved under `results/cas_frictionless/`.
See [the evaluated comparison](results/cas_frictionless/REPORT.md).
The completed declaration used 44/44 fits with zero fit failures. Independent
test RMSE was 2.4711 for the simple state hedge, 1.4057 for shallow tanh,
1.4363 for depth-two tanh, and 1.0727 for the Transformer (discounted price
units; reservoir values pool the three fresh draws' MSEs before taking the
square root). Both reservoir selections were D=32, matrix scale 1, bias
scale 1, lambda 0.1. Depth-two did not improve on shallow in this bounded
comparison. The Transformer had the lowest error, with 52.29 seconds for its
training/epoch-selection run; reservoir search totals were 6.95 and 8.63
seconds, respectively. Timing totals include different numbers of fits and
are descriptive, not a controlled hardware benchmark. The 79-test suite
passed, including the new recurrence, causal-gain and fixed-premium checks.
Transaction costs, temporary execution costs, transfer and separate reconstruction
mathematics are outside this frictionless experiment.

## CAS: bounded cost-aware reference comparison

`scripts/cas_costs.py` reuses the validated shallow/depth-two reservoirs, simple
state hedge and existing Transformer. It uses the identical Heston Asian task:
50 intervals, 4,000 / 2,000 / 8,000 training / validation / test paths, and only
time, log stock price and running Asian average as policy inputs. The saved
discounted premium **5.27387809753418** is fixed across methods, splits and cost
conditions. Historical frictionless and filtering experiments remain separate.

Let theta_k be shares held over (t_k,t_(k+1)]. Execution quantities include the
opening purchase, every rebalance, and complete terminal liquidation:

```
q_0 = theta_0
q_k = theta_k - theta_(k-1),  k=1,...,N-1
q_N = -theta_(N-1)
gain = sum_(k=0)^(N-1) theta_k * Delta S_tilde_(k+1)
cost = sum_(k=0)^N exp(-r*t_k) * S_k * (kappa*abs(q_k) + eta*q_k^2/Q_ref)
net_error = discounted_payoff - fixed_premium - gain + cost
```

The three conditions are `(kappa,eta)=(0,0)`, `(0.0005,0)`, and
`(0.0005,0.0005)`, with `Q_ref=1` share per execution. These are **illustrative,
uncalibrated** coefficients: 0.0005 is five basis points of traded notional;
the temporary term has that same rate for an order of one reference share.
At S=100, a one-share execution costs 0.05 proportional price units and,
in the combined condition, another 0.05 temporary price units before discounting.
The quadratic term is convex in each execution quantity. It is a discrete
per-execution charge with no time-step divisor, charged at entry and liquidation
as well as rebalancing. It does not change current or future simulated prices.
Turnover is reported both in shares and discounted traded notional; costs and
net errors are in discounted price units. Test errors are never recentered.

Frozen readouts minimize mean squared net error plus
`lambda/n_train * ||gamma||^2`, where `gamma = RMS_train(Psi) * beta` uses the
same training-only gain-row scaling as the fixed-premium ridge fit. The cost
objective is generally nonconvex and is not ridge. Zero-cost ridge solutions
are recomputed and checked. For positive costs, AdamW uses no weight decay,
120 full-batch steps and learning rate 0.01. Two starts are declared: ridge
initialization, and that initialization plus independent N(0,0.01^2) coefficient
perturbations in gamma coordinates using the declared seeds. Reservoir matrices
and biases stay frozen. Initial and later checkpoints are selected using
unpenalized validation net MSE; configurations use the same 12 complete tuples
and three validation reservoir draws as the frictionless shortlist.

The unchanged Transformer uses 20 epochs, batch size 256, learning rate 0.001,
zero AdamW weight decay and explicit `0.1/n_train * ||head||^2` penalty, including
head weights and bias. It uses seed 51001 at zero cost and seeds 51001/51002 at
positive costs. All competitors share the net-error objective, information and
premium. Gradient-infinity tolerance is 1e-5; reaching the fixed budget without
that criterion is reported as `budget_exhausted`, not as convergence.

The declaration caps execution at **256 counted fits**: 55 ridge fits including
fresh-candidate initializers, 144 positive-cost tuning starts, 48 positive-cost
fresh-reservoir starts, four positive-cost simple-hedge starts and five Transformer
training runs. For each positive cost, frictionless-selected and cost-selected
configurations both receive readouts trained under that same positive-cost loss.
When both selections identify the identical configuration, comparison rows
explicitly alias the same retrained fits rather than duplicate work. Actual fit
counts and aliases are saved; this does not omit a comparison or enlarge search.

From the repository root, declare with `python scripts/cas_costs.py --declare`,
then run/resume with `python scripts/cas_costs.py`, using the prepared virtual
environment and the same two-thread environment settings documented above.
See [cost-aware results](results/cas_costs/REPORT.md) for saved selections,
checkpoint histories, convergence status, failures, net errors, turnover, costs
and complete timings. Conclusions concern these implemented policy classes and
bounded optimizer runs, **not globally optimal hedging**. Cost calibration,
additional policy inputs and transfer are outside this reference comparison.

The completed cost-aware run used **220 fits**, with all 36 declared evaluation
records and zero fit failures. All conditions selected D=32, matrix/bias scale
1 and lambda 0.1 within each reservoir architecture. Thus frictionless-selected
and cost-selected configuration comparisons coincide after retraining under
each positive-cost objective; the shared fit identifiers are retained explicitly.
The overall validation-selected reservoir reference is depth-two in all three
conditions, while shallow has slightly lower fresh-draw test error. This test
observation does not change the predeclared validation selection.

Test net RMSE for zero / proportional / combined costs was 2.4711 / 2.4766 /
2.4784 for the simple hedge, 1.4057 / 1.4198 / 1.4229 for shallow, 1.4363 /
1.4509 / 1.4529 for depth-two, and 1.0727 / 1.0703 / 1.0745 for the Transformer
(discounted price units). The small Transformer difference across conditions
is not evidence that execution charges improve optimal hedging: these are
separate bounded training runs, and positive-cost validation selected seed
51002 whereas zero cost used seed 51001. Mean errors, turnover and component
costs are reported without recentering in the results table and CSV.

The optimizer audit records 43 exact ridge solves, one stationary readout run
and **176 runs reaching their step/epoch limits**. The run completed its budget;
it does not establish numerical convergence of those 176 runs. The full run
took 736.185 seconds on this environment; feature, tuning, fitting, validation
and evaluation timings are saved separately. All 86 tests passed, and an
independent audit verified all saved cash flows and selections. Historical
source/results remain preserved, and no commit was created.

## CAS: transfer and reliability

`scripts/cas_transfer.py` transfers settings between the existing Black–Scholes
Asian, Heston Asian and three-asset Heston basket Asian tasks. Model defaults,
strike 100, 50 intervals and 4,000 / 2,000 / 8,000 train / validation / test paths
are reused. Gains use discounted traded-price increments, summed over assets
for the basket. Inputs remain the existing time, log-price and running-average
channels: three channels for one asset and five for the basket. Simulated
variance, Brownian pricing controls and synthetic/filtering signals are excluded.

The documented original tuple is shallow `(D, matrix scale, bias scale, lambda)
= (64,1,1,0)` from the README quick start. Only those settings transfer from
that historical pricing example into the fixed-premium hedging formulation.
The same-cost reference is the saved overall validation-selected depth-two
`(32,1,1,0.1)` under both zero and combined costs. Each target's six-tuple list
also includes shallow `(32,1,1,0.1)`, shallow/depth-two `(16,0.5,1,0.1)` and
depth-two `(64,1,1,10)`. This is six complete tuples in total per task/condition,
not six per architecture. Original and transferred reference settings are both
included. Identical role settings share fitted policies and evaluation records.

Each task uses one independent selection partition and three validation
reservoir draws. Checkpoints and the two positive-cost starts are selected by
unpenalized validation net MSE, then the tuple is selected by the average over
those three draws. All task settings are frozen before generating any of the
nine final dataset partitions. The final study uses three independent partitions
per task and three fresh reservoir draws per partition. Final validation selects
only optimizer checkpoints/restarts; it does not select configurations again.
Preprocessing, readouts and matrices with the target input dimension are fitted
or drawn afresh. Reservoir parameters remain frozen during readout optimization.
A float32-rounded discounted training-payoff mean is frozen per partition and
shared across all methods and both conditions, without recentering errors.

Only zero costs and the validated illustrative combined condition are studied:
`kappa=eta=0.0005`, `Q_ref=1` share per asset. Charges sum over assets as well as
opening, rebalancing and liquidation dates, with no future-price modification.
The existing fixed-premium ridge and 120-step/two-start cost optimizer are reused.
The simple hedge appears in every task/condition. The pre-existing Transformer
reference comparison is repeated on Heston only, using the unchanged network,
20 epochs, one zero-cost start and two combined-cost starts; there is no
Transformer architecture search. The declaration caps counted fits at 468:
162 tuning, at most 270 final reservoir solves/starts, 27 simple-hedge fits and
nine Transformer runs. Initializer ridge solves count, including initializers
whose tuple is evaluated only under combined costs.

Methods and conditions share paths within each dataset. The same three final
reservoir draw seeds form a crossed comparison across partitions and settings.
Error bars summarize the sample standard deviation of the three dataset-level
RMSEs, each formed by first averaging three draw MSEs. They are descriptive
variation conditional on that fixed draw ensemble, not confidence intervals or
nine independent runs. Validation bars are draw standard deviations conditional
on a single selection partition. Paired transfer comparisons aggregate matched
draws within a dataset before comparing across datasets.

The runner retains one dataset and one train/validation feature group in memory,
reusing its features across costs and starts. Features are discarded after that
group; there is no persistent feature cache. Completed fit coefficients and
Transformer checkpoints are saved. Resume regeneration overhead is recorded.
Timing separates tuning, final feature generation/fitting and test deployment;
standalone per-policy costs can include shared setup, whereas actual total expense
counts unique fits and feature groups once. Memory records feature-array bytes,
boundary resident set size and the cumulative Linux process high-water mark;
these are not isolated per-fit allocation measurements. The pre-extension cost
module is archived with its original saved-study hash under
`results/cas_transfer/provenance/`.

Run from the repository root:

```bash
/workspace/.venvs/rsmc/bin/python scripts/cas_transfer.py --declare
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/rsmc-matplotlib XDG_CACHE_HOME=/workspace/.cache /workspace/.venvs/rsmc/bin/python scripts/cas_transfer.py
```

The run command reuses completed fits. A fit interrupted inside optimization
remains identified and counted; it is not silently restarted with extra budget.
Use `--report-only` to verify saved runs and regenerate the three figures and
[result tables](results/cas_transfer/REPORT.md). Conclusions concern the tested
policy classes, sparse shortlist and bounded optimization, not globally optimal
hedging. Historical experiments and reconstruction mathematics remain separate.

The completed transfer study used **360 counted fits** (cap 468), **132 unique
final evaluations** covering **186 role comparisons**, and zero failures.
All six task/condition selections chose the documented original shallow
`(64,1,1,0)` tuple, so original and task-selected roles share fitted policies.
Pooled final test net RMSE was:

| Task | Condition | Transferred reference | Original = task selected |
|---|---|---:|---:|
| Black–Scholes Asian | Frictionless | 1.255912 | 1.043560 |
| Black–Scholes Asian | Combined costs | 1.269934 | 1.062299 |
| Heston Asian | Frictionless | 1.417454 | 1.297897 |
| Heston Asian | Combined costs | 1.430741 | 1.314413 |
| Heston basket Asian | Frictionless | 1.326023 | 1.311087 |
| Heston basket Asian | Combined costs | 1.339891 | 1.325760 |

Values are discounted price units, pooling three reservoir draws within each
of three fresh datasets. The task-selected configuration had lower pooled
error in these six cells; the basket difference was small. Architecture, width
and regularization differ together, so this does not isolate an architecture
effect. On the Heston reference, the freshly fitted Transformer had lower
RMSE (1.071211 / 1.085698 for zero / combined costs) with greater fitting expense.
The simple hedge remains reported for every task and condition.

The full experiment invocation took 1,796.649 seconds, excluding subsequent
report rendering; unique fitting totaled 357.723 seconds for tuning and
1,058.423 seconds for final fits. Feature generation, deployment, artifact I/O
and reporting timings are separately saved. Peak process RSS was 1,130.06 MiB;
the largest named feature group used 381.32 MiB. These are measured wall times
and process/array memory statistics, not controlled hardware benchmarks.
Optimizer outcomes were 117 exact ridge solves, one stationary run and 242
runs reaching their limits without establishing convergence. All 91 tests
passed. The independent saved-run audit regenerated all 12 dataset partitions
and verified all residual-based CSV aggregates, selections, premiums, source
hashes and timing identities. Its reusable script and output are saved in
`results/cas_transfer/provenance/independent_audit.py` and
`results/cas_transfer/independent_audit.json`. See the report for all three
figures, descriptive dataset error bars, paired differences and per-run records.

## Battery delivery option

[Reproduce the battery study](studies/battery_delivery/README.md) or read the
[pricing and hedging results](studies/battery_delivery/REPORT.md) for a 15-minute
power option whose volume depends on battery operating history.

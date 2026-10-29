# CAS follow-up: does reservoir history help?

Protocol revised and frozen on 28 October 2026, before new test evaluation. Repository
baseline: merged PR #2, `main` at `b823b10`.
This is one frictionless follow-up to the existing three-task CAS study.
The executable protocols and seeds are in `scripts/cas_memory.py`. Generated
numerical records, plots and execution provenance are retained locally and are
excluded from version control; the verified results are summarized here.

## Question and answer

Does reservoir history improve hedging beyond an inexpensive nonlinear model
of current observable summaries? **Conclusion: Inconclusive evidence.**
Task results differ, and one shared test path dominates the Heston gap. A
performance difference does not identify its causal mechanism.

All **414 planned fits completed**: 351 selection fits and 63 final refits,
with zero failed or missing runs and no replacement seeds. The archived runner
and saved execution records place all 27 selections before test generation.
The historical CAS study
had no static tanh comparator; its outputs remain unchanged.

## Test results

MSE is in discounted currency units squared; RMSE and mean error are in
currency units. Pool squared errors across three fresh feature draws within
each partition, then across the three partitions, before taking RMSE. The SD
column is the descriptive sample SD of three partition MSEs, not a standard
error or confidence interval. Affine has one final refit per partition.

| Task | Method | Pooled MSE | Pooled RMSE | Partition MSE SD | Mean error |
| --- | --- | ---: | ---: | ---: | ---: |
| Black–Scholes | Affine | 6.1603 | 2.4820 | 0.1185 | -0.0411 |
| Black–Scholes | Static tanh | 0.9789 | 0.9894 | 0.2629 | -0.0194 |
| Black–Scholes | Reservoir | 1.1348 | 1.0653 | 0.0772 | -0.0187 |
| Heston | Affine | 5.9136 | 2.4318 | 0.0862 | -0.0643 |
| Heston | Static tanh | 12.2523 | 3.5003 | 17.6839 | -0.0506 |
| Heston | Reservoir | 1.8123 | 1.3462 | 0.1499 | -0.0486 |
| Heston basket | Affine | 4.6247 | 2.1505 | 0.1210 | -0.0259 |
| Heston basket | Static tanh | 1.4801 | 1.2166 | 0.0457 | -0.0299 |
| Heston basket | Reservoir | 1.6064 | 1.2674 | 0.0243 | -0.0200 |

Paired differences below use the same 8,000 test paths for competing methods
in each partition. Positive `static MSE − reservoir MSE` favors the reservoir.
All draws and paths are retained.

| Task | Partition 1 | Partition 2 | Partition 3 | Mean difference | Partition SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black–Scholes | +0.0579 | -0.2800 | -0.2456 | -0.1559 | 0.1859 |
| Heston | +0.4933 | +30.9278 | -0.1012 | +10.4400 | 17.7455 |
| Heston basket | -0.1149 | -0.1751 | -0.0888 | -0.1263 | 0.0442 |

Static tanh has lower pooled error in Black–Scholes and the basket; it beats
the reservoir in two of three and three of three partitions respectively.
Heston favors the reservoir in two partitions, while the third slightly favors
static tanh. On Heston partition 2, all three fresh static draws used the
validation-selected $\lambda=0$. One shared zero-payoff path (index 5298)
produced errors +196.73, −470.24 and +696.29. That path contributes 94.97% of
that partition's static squared error and 84.41% of the pooled Heston static
squared error. These are three correlated responses to one path, not three
independent tail events. The targeted audit below reproduces these outcomes
and documents extrapolation and ill-conditioned training matrices, without
finding a material implementation or numerical defect. The negative error is
profitable overhedging, not a trading loss; all three are large squared hedging
errors. This is not proof that memory caused the reservoir's aggregate
advantage. No path or draw was removed or retuned.

## Selected settings and computation

Every selected nonlinear setting has width 64, matrix scale 1 and bias SD 1.
Only the selected regularization varies below; each choice used that
partition's training and validation paths, never its test paths.

| Partition | Affine $\lambda$ | Static $\lambda$ | Reservoir $\lambda$ |
| --- | ---: | ---: | ---: |
| black_scholes_1 | 0 | 0 | 0 |
| black_scholes_2 | 10 | 0 | 0 |
| black_scholes_3 | 10 | 0 | 0 |
| heston_1 | 10 | 0 | 0 |
| heston_2 | 10 | 0 | 0 |
| heston_3 | 0.1 | 0.1 | 0.1 |
| basket_1 | 0 | 0 | 0.1 |
| basket_2 | 10 | 0.1 | 0.1 |
| basket_3 | 10 | 0 | 0.1 |

Actual component totals across all tasks, candidates, partitions and final
draws are in seconds. Feature generation includes the common passthrough;
gain construction is separate. The ridge column includes its training RMS
scaling. Common simulation/summary building took 2.107 seconds, separately.

| Method | Features | Gain rows | Ridge fits | Validation | Test evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affine | 0.179 | 0.345 | 0.051 | 0.006 | 0.004 |
| Static tanh | 43.050 | 6.072 | 6.260 | 0.092 | 0.029 |
| Reservoir | 129.582 | 5.618 | 7.261 | 0.096 | 0.028 |

The full experiment process took **216.041 seconds**, including imports,
writing results, tables and the initial figure. The internal through-save
measurement was 214.803 seconds. Peak process RSS was 660.0 MiB. Execution
used an Intel Xeon Platinum 8573C CPU, float64, two BLAS/OpenMP threads, NumPy
2.3.1, SciPy 1.17.1 and Matplotlib 3.10.9. Later figure formatting and this
editorial report are separate postprocessing. The exact executed runner and
its numerical-module hashes are archived locally. The later history-window
extension changed source files without rewriting those historical hashes.

Across tasks the static method's full selection cost was 26.98 seconds versus
69.13 seconds for the reservoir. Mean single-draw inference on an 8,000-path
batch, including features and gain construction, was 0.535/1.906 seconds
(static/reservoir) in Black–Scholes, 0.665/1.098 in Heston and 0.632/1.703 in
the basket. Width and candidate counts were matched; computation was measured
and was not equal. Per-partition training, feature, solve, validation and
inference timings remain in the local machine-readable artifacts.

## Information and current summaries

Retain the existing risk-neutral models, defaults, strike 100, maturity
$T=1$, rate $r=0.03$, and uniform grid $N=50$. Let
$\mathcal F_k^{obs}=\sigma(S_0,\ldots,S_k)$, with deterministic model settings
known. A current summary can itself contain a running payoff statistic; the
question concerns history **beyond** that summary.

For both scalar tasks define
$$
A_k=\frac{1}{k+1}\sum_{j=0}^k S_j,\qquad
x_k=(t_k/T,\log(S_k/S_0),\log(A_k/S_0))\in\mathbb R^3.
$$
For the basket let $B_k=(S_k^1+S_k^2+S_k^3)/3$ and
$\bar B_k=(k+1)^{-1}\sum_{j=0}^k B_j$, and use
$$
x_k=(t_k/T,\log(S_k^1/S_0^1),\log(S_k^2/S_0^2),
       \log(S_k^3/S_0^3),\log(\bar B_k/B_0))\in\mathbb R^5.
$$
Thus all current constituent prices are available; the current basket is
recoverable from them. No separate constituent running averages are needed for
this equal-weight basket payoff. These are exactly the CAS adapter's channels.
The optional variance, noisy-signal and filtering inputs elsewhere in the
repository were not declared for these CAS tasks and are excluded, as are
future observations, lag vectors and new estimated signals.

The payoff averages dates 1 through $N$, whereas these summary averages
include inception. There is no information loss: the accrued sum excluding
inception is $(k+1)A_k-S_0$, or $(k+1)\bar B_k-B_0$.
The existing log coordinates are invertible given initial prices. Here
“affine” means affine in these coordinates, not in raw prices.

## Three policy classes and common objective

Let $d=\dim x_k$, $u_k=(1,t_k,x_k^2,\ldots,x_k^d)$, and fix
$D=64$. All methods share this intercept and current-summary passthrough.
For each traded stock $a$, holdings at date $k<N$ are
$\theta_k^a=(\beta^a)^\top\phi_k$, with one coefficient vector shared
across dates:

1. **Affine summaries:** $\phi_k=u_k$.
2. **Static frozen tanh:**
   $$
   z_k=\tanh(Ux_k+c),\qquad \phi_k=(u_k,z_k),\qquad
   U_{ij}\sim N(0,s^2/d),\quad c_i\sim N(0,1).
   $$
   Draw parameters once and freeze them across train/validation/test. This map
   has no recurrent state, lagged inputs or trained hidden weights: identical
   summaries always produce identical features for a fixed draw.
3. **Existing shallow reservoir:**
   $$
   h_0=0,\qquad
   h_{k+1}=h_k+\sum_{r=1}^d\tanh(W_rh_k+b_r)
                          (x_{k+1}^r-x_k^r),\qquad \phi_k=(u_k,h_k),
   $$
   with frozen $(W_r)_{ij}\sim N(0,s^2/D)$ and
   $(b_r)_i\sim N(0,1)$. Hold $\theta_k$ before observing the next
   increment. Retain the implemented recurrence without modification.

There is no empirical input centering/whitening in any method. The only fitted
preprocessing is the training gain-row scaling below. Driver time is $t/T$
and passthrough time is $t$, as historically; $T=1$ here. The common
readout **already matches historical CAS** `[1,t,current non-time summaries,h]`.
The older pricing helper `[1,t,h]` is a different pipeline. Neither historical
readout nor historical results are retroactively changed.

Each nonlinear method has 68 trained coefficients in a scalar task and
$3\times70=210$ in the basket; affine has 4 and 18 respectively. Width and
readout fitting are matched, not frozen parameter count or runtime. Static
features cost roughly $O(NdD)$ per path versus $O(NdD^2)$ for the dense
reservoir. The two scale parameters also have different dynamical effects.

With discounted payoff $H$, use the common fixed premium
$p=\operatorname{float32}(\operatorname{mean}_{train}H)$ separately in each
partition. Define
$$
\widetilde S_k=e^{-rt_k}S_k,\quad
G(\theta)=\sum_{k=0}^{N-1}\sum_a\theta_k^a
                 (\widetilde S_{k+1}^a-\widetilde S_k^a),\qquad
e=H-p-G(\theta).
$$
The target is $E[e^2]$. Trading uses discounted traded prices, not simulated
Brownian increments. Costs are zero; terminal liquidation is free and is
consistent with this terminal-wealth accounting.

Form asset-major gain rows
$\Psi_i^{a,j}=\sum_k\phi_{i,k}^j\Delta\widetilde S_{i,k+1}^a$.
Set $q_j=\sqrt{\operatorname{mean}_{train}\Psi_j^2}$, replacing values below
$10^{-12}$ by 1, and $Z=\Psi/q$. Reuse the existing augmented least-squares
solver for
$$
\widehat\gamma=\arg\min_\gamma
 \sum_{i\in train}(H_i-p-Z_i\gamma)^2+\lambda\|\gamma\|_2^2,
 \qquad \widehat\beta=\widehat\gamma/q.
$$
The mean-loss penalty is $\lambda/4000$. Penalize every holding coefficient,
including the policy intercept; that intercept is a constant holding, not
fitted initial capital. Do not center gain rows, targets or test errors. Carry
training scales unchanged into validation/test. No optimizer restarts apply.

## Frozen selection protocol and exact budget

For **each nonlinear method**, use this complete ordered candidate list;
dimension 64 and bias standard deviation 1 are fixed throughout:

| Candidate | Weight scale $s$ | Regularization $\lambda$ |
| --- | ---: | ---: |
| 0 | 0.5 | 0 |
| 1 | 0.5 | 0.1 |
| 2 | 0.5 | 10 |
| 3 | 1 | 0 |
| 4 | 1 | 0.1 |
| 5 | 1 | 10 |

Affine candidates are $\lambda=(0,0.1,10)$, in that order. Do not create
duplicate affine draws to inflate its budget. This is a bounded comparison,
not a claim that either nonlinear family has been exhaustively tuned.

Select independently **within each of three dataset partitions per task**;
there is no additional selection partition. All six tuples remain unchanged.
For task index $a=0,1,2$ in the table's order and partition $b=1,2,3$, use
dataset seed `2026102800 + 100*a + b`. Each partition has
4,000/2,000/8,000 independent train/validation/test paths from the existing
adapter's child streams. There are nine partitions and 72,000 unique test paths.
All three methods share exactly the same paths within each partition.

For $m=0$ static, $m=1$ reservoir and draw $j=1,2,3$, tuning seeds are
`2026103800 + 1000*a + 100*b + 10*m + j`; final seeds add 10,000.
Within each method/partition/draw, reuse base Gaussian parameters across scales
and penalties; draw matrices before biases using `default_rng`.

Select separately for each method and partition using unpenalized validation
MSE, averaged across three tuning draws. Exact ties use candidate order.
Freeze all **27** choices before generating any test child stream. Refit the
chosen settings on the same training partition using three fresh draws per
nonlinear method and an explicit affine refit. Do not combine train and
validation or select final draws. This evaluates average single-draw
performance, not an ensemble of averaged holdings.

| Stage | Calculation | Ridge fits |
| --- | --- | ---: |
| Nonlinear selection | 9 partitions × 2 methods × 6 candidates × 3 draws | 324 |
| Affine selection | 9 partitions × 3 penalties | 27 |
| Nonlinear final | 9 partitions × 2 methods × 3 fresh draws | 54 |
| Affine final refit | 9 partitions | 9 |
| **Total** | **351 selection + 63 final** | **414** |

The declaration was saved locally in `plan.json`. Test paths were not used by previous
CAS runs. Train/validation are generated first and regenerated with the test
child after selections freeze; 126,000 unique paths correspond to 180,000
actually generated paths, with regeneration charged in timing.

This is the entire planned fit budget. Record failures; do not replace seeds,
expand the candidate list or average only survivors. An incomplete selection
grid or missing final fit leaves the comparison incomplete. No costs,
depth-two model, Transformer, new task or extra tuning stage is included.

## Interpretation by model

Matching performance would not prove equivalence. A reservoir advantage
supports usefulness relative to this comparator and budget; representation,
regularization and estimation also differ, so the gap does not isolate memory.

In exact-grid Black–Scholes with fixed volatility, current price and the accrued
average are Markov sufficient for the payoff and future prices. Under this
frictionless martingale quadratic objective, an unrestricted optimal hedge can
depend on these summaries. A recurrent advantage could therefore reflect
finite-class approximation rather than missing state information. Discrete
trading still permits residual hedging error.

The Heston simulators use projected variance Euler and log-price updates.
Variance is latent to policies. Earlier prices on this observation grid may
inform its conditional distribution beyond current prices and averages. This
creates scope for useful memory but does not establish that the reservoir
extracts it. The statement concerns these discretized, price-only simulations,
not continuously observed Heston, where quadratic variation changes available
information.

## Targeted primary-source check

The following full-text passages were read, beyond their abstracts. None
resolves the proposed finite-budget comparison.

1. **Bühler, Gonon, Teichmann and Wood, _Deep Hedging_ (2019):**
   [full text](https://arxiv.org/pdf/1802.03042), §2 and §4.2,
   equation (4.1), Remark 4.6 and Proposition 4.9. Strategies may use observable
   history; appropriate Markov structure permits reduced state dependence.
   The approximation result concerns expanding trained neural-network classes,
   not a finite-width reservoir/static-feature ranking. Its §5 Heston setup
   includes variance information, unlike this price-only comparison. The
   Black–Scholes interpretation above additionally uses this repository's
   payoff state and martingale quadratic objective.
2. **Rahimi and Recht, _Weighted Sums of Random Kitchen Sinks_ (2008):**
   [full text](https://proceedings.neurips.cc/paper_files/paper/2008/file/0efe32849d230d7f53049ddc4a4b0c60-Paper.pdf),
   Algorithm 1, equations (3)–(6), Theorem 1 and §2. Frozen random nonlinear
   features with fitted linear coefficients are an established inexpensive
   comparator; tanh satisfies the bounded-feature premise. Their theorem uses
   a specified function class, coefficient bounds and Lipschitz classification
   loss. It is not directly a guarantee for unbounded traded-gain rows and
   squared hedging errors.
3. **Cuchiero, Gonon, Grigoryeva, Ortega and Teichmann,
   _Discrete-time Signatures and Randomness in Reservoir Computing_ (2022):**
   [full text](https://arxiv.org/pdf/2010.14615), §II/Theorem II.4 and
   §III/Theorems III.7–III.8. The prescribed random state-affine reservoirs
   approximate suitable history-dependent filters with linear readouts under
   bounded-input and regularity assumptions. Those constructions differ from
   the implemented increment-driven tanh recurrence. Their results neither
   show that this hedge needs additional history nor predict a winner here.

## Implementation, checks and artifacts

### Independent targeted review, 28 October 2026

The review checkout contained the implementation and completed outputs. A read-only
review confirmed observable payoff summaries, stateless tanh features, causal
reservoir holdings, matched passthrough/readout sizes, training-only gain
scaling and common premium/accounting. All **43 focused tests passed again**.
Independent recomputation of selection, residual aggregation and timing tables
passed 1,606 numerical comparisons (maximum discrepancy $2.22\times10^{-16}$).

The saved plan and archived runner agree on **324 + 27 + 54 + 9 = 414 fits**;
every planned tuple/seed is present and completed. All 135 candidate means and
27 choices reproduce. Nineteen execution-source hashes match. The archived
runner persists every choice and its hash before opening test streams; ordered
fit/group records, nine train/validation-only generations followed by nine
full generations, and the log corroborate this sequence. This verifies the
available execution evidence, not an immutable record of every possible
off-record access. Nine test hashes are distinct from historical CAS hashes.

For **Heston partition 2, zero-based test path 5298**, regeneration matches all
three saved split hashes. The simulated spot/variance recurrence reproduces
exactly; variance is inspected only for this audit and never enters a policy.
Spot ends at 13.9916, the payoff average over dates 1–50 is 41.6471, payoff is
zero and the fixed premium is 5.4320. Over trading dates 0–49, this path's
price/average ranges are 15.141–101.128 / 43.367–100.564, compared with
training ranges 28.846–201.673 / 56.478–142.984. Its price or average leaves
the training-wide marginal range at dates 32–49 (the same-date range from
date 8). Thus the excursion is present in the declared current inputs.

The static final seeds are `2026115001`–`2026115003`, in row order:

| Draw suffix | Path holdings range (shares) | Discounted gain | Smallest singular value of $Z$ | Condition number of $Z$ |
| --- | ---: | ---: | ---: | ---: |
| 5001 | -1.398 to 44.227 | -202.1588 | $7.4852\times10^{-8}$ | $6.1994\times10^9$ |
| 5002 | -129.489 to 0.557 | +464.8108 | $3.2182\times10^{-7}$ | $1.4711\times10^9$ |
| 5003 | 0.040 to 120.651 | -701.7210 | $5.1782\times10^{-7}$ | $9.3273\times10^8$ |

The corresponding training holdings stay within -5.463 to 3.342 shares.
The $4000\times68$ normalized training matrices have largest singular values
464.040, 473.429 and 482.984; all 68 modes exceed the actual solver cutoff.
Raw coefficient norms are $1.6922\times10^6$, $4.3671\times10^5$ and
$6.2702\times10^5$; normalized coefficient norms are $3.1453\times10^7$,
$4.5125\times10^6$ and $1.0194\times10^7$. These magnitudes alone do not
establish the cause. A singular-vector residual diagnostic for possible
solve error at this path is below 0.00037 currency units in every draw;
it is a finite-precision diagnostic, not a certified bound.

All three full static error vectors reproduce exactly with saved coefficients.
Direct holdings-times-increments, a nominal self-financing cash account with
liquidation, and extended-precision evaluation agree with the saved path
errors within $5\times10^{-8}$ currency units. For example, draw 5003 holds
55.837 shares at date 36 before a discounted price change of -3.03547,
contributing -169.492 to its gain. Affine/reservoir path errors also reconcile;
they are -19.389 and -22.887/-10.613/-15.835, respectively.

**Finding:** these are valid realized outcomes of the fitted strategies;
no material defect requiring invalidation or correction was found. The
diagnostics support extrapolation sensitivity of the unregularized static
fits. The relative roles of finite training coverage, representation geometry
and conditioning remain unresolved; the reservoir's memory mechanism is not
identified. No new fits, selection, replacement draws or path exclusions were
used. Another experiment is not required to report this bounded comparison's
mixed, inconclusive answer, and this audit supplies no reason to repeat it
to seek a favorable result.

`rsmc/hedging.py` adds only the frozen static-tanh map and its gain-row adapter.
The shallow recurrence is unchanged. The single runner `scripts/cas_memory.py`
reuses the existing simulation, summaries, common readout, gain construction,
fixed-premium ridge solver, evaluation and CSV utilities.

**43 focused tests passed**, including seven new cases for statelessness under
changed history with identical summaries, frozen parameters, matching current
passthrough, future-input causality, training-only normalization and nominal
cash accounting with interest and full liquidation for one and three stocks.

An independent artifact check recomputed all 135 candidate scores and 27
selections exactly. It checked 63 residual vectors, 27 dataset rows, nine
summary rows and nine paired comparisons: 369 numerical comparisons, maximum
absolute discrepancy $3.55\times10^{-15}$. Nine test-price hashes are mutually
distinct and disjoint from all 13 distinct historical CAS test hashes found.
All 228 historical artifact hashes are unchanged. A separate replay of the
three Heston partition-2 static models used saved coefficients and regenerated
existing paths, with no new fits; all saved errors matched exactly. This
verification regeneration is excluded from experimental timing/path counts.

The excluded local artifacts are `plan.json`, `results.json`, `summary.csv`,
`dataset_summary.csv`, `paired.csv`, nine compressed test-residual archives,
`comparison.png`, `run.log`, `timing.json` and the executed-source snapshot.
`results.json` retains selections, fit statuses, final coefficients and
training gain scales, source/path hashes, environment and component timings.
There are 36 affine fits and 189 fits for each nonlinear method. No failures
were hidden and no configuration, seed or test path was replaced.

The focused pytest command covered `test_memory_hedging`, `test_cas_components`,
`test_hedging`, `test_transfer_tasks` and `test_reservoir`, using `python -m pytest`
in the recorded environment with two BLAS/OpenMP threads.

The completed study needs no resume. The current source guard intentionally
rejects historical `--report-only` processing when numerical-source hashes
differ. Those historical outputs require their matching archived sources;
changing saved hashes is not a valid workaround.

Three partitions and three draws per selected setting provide limited evidence.
The fixed width, narrow scale/penalty shortlist, one grid and simulated model
laws limit generalization. Equal widths and RMS-scaled ridge penalties do not
make the two representation classes or regularization geometries equivalent.
Matching performance would not prove equivalence; an advantage only concerns
this comparator and budget. The study does not establish indispensable history,
superiority over every nonlinear current-summary model, global optimality or
universal robustness.

## History-window extension — 28 October 2026

**All 756 declared fits completed: 648 selection fits and 108 final fits.**
The reservoir has lower draw-averaged MSE than both fixed window models in all
nine partitions.
Current-summary tanh has the lowest pooled MSE on every task and beats the
reservoir in eight partitions. These fresh datasets support the reservoir over
the tested windows, without demonstrating a benefit from additional history.
Historical results above remain unchanged.

### Fixed design and implementation

The four fixed representations use 64 nonlinear features. Window inputs are
$[x_k,x_{\max(k-1,0)},\ldots,x_{\max(k-w,0)}]$: windows 0/5/20 contain 1/6/21
existing summary vectors. Their running averages already retain earlier payoff
information. Only the nonlinear input expands; current-summary passthroughs,
observable inputs and the reservoir recurrence remain unchanged. Window zero
reproduces the existing static map.

Static/window weights have Gaussian SD $s/\sqrt{d(w+1)}$; reservoir channel
matrices retain SD $s/\sqrt{64}$. Bias SD is 1. Each representation separately
selects among the same six $(s,\lambda)$ tuples, scales 0.5/1 and penalties
0/0.1/10, by mean validation MSE across three draws; exact ties use candidate
order. Windows are never selected against one another.

Three fresh partitions per task each contain 4,000/2,000/8,000 training,
validation and test paths over 50 intervals, shared across methods. All 36
settings froze before test generation; three fresh draws per setting were
refitted on the same training paths. The budget is
$9\times4\times6\times3=648$ selection fits plus $9\times4\times3=108$ final
fits. No training/validation merge or averaged holdings is used.

For task index $a=0,1,2$ in Black–Scholes, Heston and basket order, partition
$b=1,2,3$, method index $m=0,1,2,3$ in current-summary/window-5/window-20/reservoir
order, and draw $j=1,2,3$, dataset seeds are `2026200000 + 100*a + b`, tuning
seeds are `2026210000 + 1000*a + 100*b + 10*m + j`, and final seeds are
`2026220000 + 1000*a + 100*b + 10*m + j`. These 225 seeds were unused before
the extension. The local `windows/plan.json` and `windows/declaration.json`
froze the complete protocol and source hashes before execution; the runner
contains the same model defaults, candidates, seed rules and conventions.
The training-derived premium, deterministic inputs, training-only gain RMS,
ridge objective and discounted-stock gains remain as defined above. All methods use 2,000-path
batches retaining complete histories.

### Errors and variation

MSE is in discounted currency squared; RMSE is in currency. Squared errors are
averaged across draws within each partition, then equally across the three
partitions. Partition SD is the sample SD of those three MSEs. Draw SD is the
mean of the three within-partition sample SDs of draw MSEs. They describe
different sources of variation; neither is a confidence interval.

| Task | Representation | MSE | RMSE | Partition MSE SD | Mean within-partition draw MSE SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Black–Scholes | Current tanh | 0.9237 | 0.9611 | 0.1568 | 0.0914 |
| Black–Scholes | Window 5 | 1.2353 | 1.1114 | 0.0686 | 0.0488 |
| Black–Scholes | Window 20 | 1.8009 | 1.3420 | 0.0405 | 0.0509 |
| Black–Scholes | Reservoir | 1.0842 | 1.0413 | 0.0204 | 0.0494 |
| Heston | Current tanh | 1.4209 | 1.1920 | 0.0696 | 0.0707 |
| Heston | Window 5 | 1.7905 | 1.3381 | 0.0741 | 0.0707 |
| Heston | Window 20 | 2.2078 | 1.4859 | 0.0092 | 0.0319 |
| Heston | Reservoir | 1.6897 | 1.2999 | 0.0746 | 0.0901 |
| Basket | Current tanh | 1.5517 | 1.2457 | 0.1667 | 0.2001 |
| Basket | Window 5 | 1.7208 | 1.3118 | 0.0194 | 0.0431 |
| Basket | Window 20 | 2.1084 | 1.4520 | 0.0526 | 0.0884 |
| Basket | Reservoir | 1.6395 | 1.2804 | 0.0501 | 0.0418 |

Paired differences use shared test paths and average feature draws within
each partition. Positive values below favor the reservoir. All six pairwise
comparisons, including window versus current-summary models, are retained in
the excluded local `windows/paired.csv`.

| Task | Current − reservoir | Window 5 − reservoir | Window 20 − reservoir |
| --- | ---: | ---: | ---: |
| Black–Scholes | -0.1605 ± 0.1370 | +0.1511 ± 0.0575 | +0.7167 ± 0.0506 |
| Heston | -0.2688 ± 0.1329 | +0.1008 ± 0.0560 | +0.5180 ± 0.0734 |
| Basket | -0.0877 ± 0.1171 | +0.0814 ± 0.0384 | +0.4689 ± 0.0201 |

Entries are mean difference ± descriptive partition SD.
Every reservoir draw beats every window-20 draw within its partition. Reservoir
and window-5 draw ranges overlap in Black–Scholes partition 2, Heston partitions
2–3 and basket partition 1; the smaller window-5 advantage is less consistent
across draws.

All selected scales are 1 except current-summary tanh in Black–Scholes
partition 1 and Heston partition 2, which select 0.5. All penalties are zero
except reservoir in all basket partitions, window 20 in basket partition 1,
and window 5 in basket partition 3, which select 0.1. All settings and
candidate validation scores remain in the excluded local `windows/results.json`.

**Tail concentration:** every path and draw is retained. In Black–Scholes
partition 2, current tanh's largest path contributes 15.97% of its squared
error, and the largest five contribute 30.92%. Basket partition 1's
current-minus-reservoir mean is +0.0467, while its five largest absolute
pathwise differences contribute +0.3158, offset by other paths. Its window-5
comparison is +0.0408 with a +0.0281 five-path contribution. Conversely, Heston
partition 3's window-5-minus-reservoir mean is +0.0617 despite a -0.1239
five-path contribution. These are concentration diagnostics, not trimmed
estimates; signed contributions can exceed net differences through cancellation.
Window 5 beats current tanh only in basket partition 1, by 0.0058 MSE; current
tanh's path 7436 alone contributes +0.2324 to that difference. This small reversal
is tail-sensitive. The pooled reservoir-versus-window advantages are not carried
by a few favorable extreme paths.
All 90 method/pair diagnostics and path indices remain in the local results.

### Computation, parameters and checks

Complete process wall time was **466.173 seconds**, including imports, saving,
aggregation and the figure; internal through-save time was 464.481 seconds.
Peak RSS was 477.1 MiB. Execution used the same pinned Python environment,
float64, two BLAS/OpenMP threads and an Intel Xeon Platinum 8573C CPU.
Common simulation and summary construction took 3.282 seconds.

| Representation | Features (s) | Gain rows (s) | Fits (s) | Validation (s) | Test evaluation (s) | Full selection (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current tanh | 41.669 | 5.685 | 7.931 | 0.133 | 0.037 | 30.036 |
| Window 5 | 56.213 | 5.597 | 7.415 | 0.104 | 0.038 | 34.676 |
| Window 20 | 110.858 | 6.365 | 7.640 | 0.092 | 0.648 | 56.829 |
| Reservoir | 130.851 | 5.784 | 8.896 | 0.092 | 0.028 | 76.806 |

Component columns cover all fits and final evaluations. Selection is a subset
total, not an additional cost. Parameter draws total 0.184 seconds; lag
construction is included in feature time. Complete wall time also includes
common simulation and artifact overhead. Mean single-draw inference on 8,000
paths, including features/gains, is 0.512/0.652/1.259/1.463 seconds in
representation order. These are one-run measurements with unestimated scheduling
variability. Equal width does not imply equal computation.

All representations fit **68 readout coefficients** in scalar tasks and **210**
in the basket. Frozen parameter counts per draw are:

| Representation | Scalar frozen parameters | Basket frozen parameters |
| --- | ---: | ---: |
| Current tanh | 256 | 384 |
| Window 5 | 1,216 | 1,984 |
| Window 20 | 4,096 | 6,784 |
| Reservoir | 12,480 | 20,800 |

Counts exclude the shared premium and preprocessing statistics; draws are
separate models, not an ensemble.

**55 relevant tests passed**, including zero-window equivalence, lag order and
padding, causality, scalar/basket alignment and direct hedge accounting. An
independent audit reproduced 216 candidate scores, 36 choices, 108 residual
vectors' metrics, all tables, tail diagnostics, timings and parameter counts:
2,302 numerical comparisons, maximum discrepancy $6.66\times10^{-16}$. All
21 source hashes match; the selection-freeze timestamp precedes test generation.
Nine new test hashes are distinct and disjoint from 22 historical hashes.
Saved-coefficient replays checked 540 representative model/path evaluations,
including every model's largest errors. Nominal cash and discounted gains agree
within $3.84\times10^{-13}$. Reordered arithmetic changes some errors by a few
millionths; replaying the most sensitive case with the original batching
reproduces its full saved error vector exactly.

**Zero model failures, interrupted fits or replacement draws.** A timing
launcher failed before Python started; the local execution log retains the
failure and subsequent successful run. The failed launcher performed no fits.

Historical numerical artifacts, manifests and hashes remain unchanged in the
local archive. Pre-extension hedging source and executed runners are retained
there as provenance. Historical source guards remain intact. Generated
extension artifacts are excluded local outputs under `windows/`.

### Interpretation and limits

The reservoir beats the declared window maps on pooled error, with higher
selection cost. Its measured inference cost exceeds window 5 on every task,
but is slightly below window 20 on Heston. The cheaper current-summary model
has lower pooled error on every
task, so additional history has no demonstrated benefit here. The window
comparison does not isolate a memory mechanism: the reservoir can retain longer
history, while input dimension, representation geometry and regularization
also differ. Adding lags at fixed width need not improve approximation.

In the simulated Black–Scholes model, time, current price and the running average
already determine the conditional law relevant to the remaining payoff. Any
history-map advantage would concern finite-model performance, not additional
population information. Latent Heston variance does not by itself establish
that extra history improves the fitted hedge.

Three partitions, three draws per setting, one grid and a narrow shortlist
limit generalization. Shared paths and feature draws are not independent
dataset replications. No claim of indispensable history or superiority over
all nonlinear history models follows.

Both studies are complete. To reproduce either protocol in a fresh Linux
checkout, first activate an environment with the repository dependencies,
then declare the protocol before running it from the repository root:

```bash
python scripts/cas_memory.py --declare
python scripts/cas_memory.py
python scripts/cas_memory.py --windows --declare
python scripts/cas_memory.py --windows
```

These commands execute the declared 414-fit and 756-fit studies respectively;
they are not needed to read the results above. For an existing completed local
window run with matching frozen source hashes, use
`python scripts/cas_memory.py --windows --report-only` to rebuild its tables and
figure without fits. Do not use the ordinary run command for completed outputs:
it records another invocation and replaces the latest invocation timing.
A fresh checkout does not contain the numerical inputs for `--report-only`.

## Optional two-factor Heston benchmark — implementation only

**Implementation record before execution; completed results follow below.** This separate mode
asks whether historical representations help hedge when observed total variance
has slow and fast components. It is a finite-factor stochastic-volatility model,
not rough volatility. Unequal reversion speeds make the conditional evolution of
total variance depend on its hidden composition. This motivates the comparison;
it does not establish that history improves the optimal Asian hedge.

One stock has two independent Brownian pairs, with within-pair correlations
$\rho=(-0.7,-0.7)$. Parameters are $S_0=K=100$, $r=0.03$, $T=1$,
$v_0=\theta=(0.02,0.02)$, $\kappa=(0.5,5)$ and $\xi=(0.3,0.3)$.
With $\Delta=1/50$, the numerical target is
$$
v_{i,k+1}=\left[v_{i,k}+\kappa_i(\theta_i-v_{i,k})\Delta
             +\xi_i\sqrt{v_{i,k}\Delta}\,Z^W_{i,k+1}\right]^+,
$$
$$
\log S_{k+1}=\log S_k+\left(r-\tfrac12 U_k\right)\Delta
       +\sum_{i=1}^2\sqrt{v_{i,k}\Delta}\,Z^B_{i,k+1},
\qquad U_k=v_{1,k}+v_{2,k}.
$$
Both stock drift and diffusion use current variances. Separate projections need
not collapse pathwise to the ordinary Heston scheme when factor parameters match.
The Asian payoff, discounting, training-derived fixed premium and ridge objective
remain those defined above.

All methods observe exactly
$x_k=[t_k/T,\log(S_k/S_0),\log(A_k/S_0),z_k]$, with
$A_k=(k+1)^{-1}\sum_{j=0}^kS_j$ and $z_k=U_k/0.04$.
They receive neither individual factors nor innovations. The five fixed
representations are current tanh, window-5 tanh, window-20 tanh, two-EWMA tanh,
and the unchanged shallow reservoir. Window order and inception padding are
unchanged. For each rate $\kappa\in\{0.5,5\}$, the EWMA starts at $q_0=z_0$
and updates $q_k=e^{-\kappa\Delta}q_{k-1}+(1-e^{-\kappa\Delta})z_k$.
These historical summaries are not exact factor estimates. Only the EWMA
nonlinear map receives $[x_k,q_k^{0.5},q_k^5]$; every readout retains
$[1,t_k,\log(S_k/S_0),\log(A_k/S_0),z_k]$ plus 64 nonlinear features.
Thus all methods fit 69 coefficients. Tanh input dimensions are 4/24/84/6,
with Gaussian weight SD $s/\sqrt{\text{input dimension}}$; reservoir channel
matrices retain SD $s/\sqrt{64}$. Every bias SD is 1.

There are three partitions, each with 4,000/2,000/8,000 training/validation/test
paths. Each method separately selects the existing six $(s,\lambda)$ tuples
from $s\in\{0.5,1\}$, $\lambda\in\{0,0.1,10\}$, using mean validation MSE
across three draws and candidate-order ties. All 15 selections freeze before
test generation; three fresh final draws per selection are fitted on training
paths only. The budget is $3\times5\times6\times3=270$ selection fits plus
$3\times5\times3=45$ final fits, totaling **315**. The existing batching,
accounting, timing, failure, aggregation and resume logic is shared.

For partition $b=1,2,3$, method index $m=0,\ldots,4$ in the order above,
and draw $j=1,2,3$, dataset seeds are `2026300000+b`, tuning seeds are
`2026310000+100*b+10*m+j`, and final seeds are `2026320000+100*b+10*m+j`.
The 93 seeds are disjoint from the historical protocols. Declaration checks
available prior JSON records for collisions, saves the plan and source hashes,
and generates no paths. Historical files are not required in a fresh checkout.

**Fixed-grid martingale and moment justification.** Conditional on the full
history, $\sum_i\sqrt{v_{i,k}\Delta}Z^B_{i,k+1}$ is centered Gaussian with
variance $\Delta U_k$. Its exponential moment gives
$E[\widetilde S_{k+1}\mid\mathcal F_k]=\widetilde S_k$ exactly for the
declared transition law. The tower property gives the same identity in the
observed price/total-variance filtration; within-pair leverage does not alter it.

For each projected factor, write $v'=(b+cv+\xi\sqrt{\Delta v}Z)^+$,
where $b=\kappa\theta\Delta$, $c=1-\kappa\Delta$. For every $a\ge0$,
$$
E[e^{av'}\mid v]\le1+\exp\left(ab+
           [ac+\tfrac12a^2\xi^2\Delta]v\right).
$$
Replacing the coefficient of $v$ by its positive part gives a bound by a
positive exponential moment. Starting from deterministic factors, induction
proves all positive exponential moments of each factor at every finite grid
date; Hölder gives the same for $U_k$. For every $p\ge1$,
$$
E[S_{k+1}^p]
\le e^{pr\Delta}\sqrt{E[S_k^{2p}]
                       E[e^{p(p-1)\Delta U_k}]}.
$$
Simultaneous induction over dates for all $p$ proves finite stock moments,
including second moments, while allowing stock/variance dependence. Jensen's
inequality gives square integrability of the Asian payoff. This is an analytic
finite-grid argument, not a simulation check or a uniform grid-refinement claim.

On Linux, with the repository environment activated, declare with
`python scripts/cas_memory.py --two-factor --declare`, then run or resume with
`python scripts/cas_memory.py --two-factor`. Outputs go to
`results/cas_memory/two_factor/`. `--two-factor` and `--windows` are mutually
exclusive. Source checks include the new module and all other `rsmc` modules.
Historical declarations, hashes and numerical outputs remain unchanged and
require their original source versions; the new mode does not bypass those checks.

## Two-factor benchmark — completed 29 October 2026

**315/315 fits completed: 270 selection and 45 final fits, with zero failures,
interrupted fits or replacement draws.** The declaration froze the specified
model, four observable summaries, five representations, six candidates and 93
seeds before execution. All 15 validation selections froze before test generation.
Three partitions each use 4,000/2,000/8,000 paths and 50 intervals; each selected
representation has three fresh final draws. No old study was rerun or retuned.

MSE is in discounted currency squared and RMSE in currency. Squared errors are
averaged over feature draws within partitions, then equally over three partitions.
Partition SD describes those three means. The last three columns report each
partition's draw-averaged MSE ± within-partition draw SD. Neither SD is a confidence
interval; shared paths and feature draws are not independent datasets.

| Representation | Pooled MSE | RMSE | Partition MSE SD | Partition 1 MSE ± draw SD | Partition 2 | Partition 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current tanh | 2.1195 | 1.4559 | 0.6006 | 2.7915 ± 1.2082 | 1.6348 ± 0.1046 | 1.9323 ± 0.1506 |
| Window 5 | 2.4300 | 1.5588 | 0.0932 | 2.5133 ± 0.0855 | 2.3294 ± 0.0282 | 2.4472 ± 0.0380 |
| Window 20 | 3.0628 | 1.7501 | 0.1134 | 3.1454 ± 0.0532 | 2.9335 ± 0.0596 | 3.1094 ± 0.0861 |
| Two-EWMA tanh | 1.9926 | 1.4116 | 0.0617 | 2.0076 ± 0.0451 | 1.9248 ± 0.0554 | 2.0454 ± 0.0843 |
| Reservoir | 2.5166 | 1.5864 | 0.1821 | 2.4788 ± 0.1041 | 2.3564 ± 0.1595 | 2.7146 ± 0.3117 |

Paired differences use the same test paths, averaging draws first. Positive
values favor the method on the right. All ten comparisons per partition are
saved locally in `two_factor/paired.csv`; the question's main comparisons are:

| MSE difference | Mean ± partition SD |
| --- | ---: |
| Current − window 5 | -0.3104 ± 0.5176 |
| Current − window 20 | -0.9432 ± 0.5140 |
| Current − EWMA | +0.1270 ± 0.5758 |
| Current − reservoir | -0.3971 ± 0.6154 |
| Window 5 − reservoir | -0.0866 ± 0.1596 |
| Window 20 − reservoir | +0.5462 ± 0.1385 |
| EWMA − reservoir | -0.5240 ± 0.1273 |

All settings have D=64 and bias SD 1. Selected `(scale, ridge penalty)` tuples:

| Representation | Partition 1 | Partition 2 | Partition 3 |
| --- | --- | --- | --- |
| Current tanh | (1, 0) | (1, 0) | (1, 0) |
| Window 5 | (1, 0) | (0.5, 0) | (1, 0) |
| Window 20 | (0.5, 0) | (0.5, 0) | (0.5, 0) |
| Two-EWMA tanh | (1, 0) | (1, 0) | (1, 0) |
| Reservoir | (0.5, 0.1) | (0.5, 0.1) | (0.5, 0.1) |

**Extreme errors are retained.** Current tanh's partition-1 path 2285 contributes
26.90% of that partition's draw-averaged squared error; its largest five paths
contribute 40.50%. Draw 2026320102 has signed error -130.1814 on path 2285.
This path alone contributes +0.2447 to pooled current-minus-EWMA MSE, exceeding
the net +0.1270; the five largest absolute paired contributions sum to +0.3934.
Thus the pooled EWMA improvement is strongly tail-sensitive. These are signed
contributions, which can exceed the net difference through cancellation; no
paths were removed and no trimmed estimates were substituted.

Reservoir partition 3 also has concentrated errors: path 5671 contributes 8.60%
of its squared error, and the largest five contribute 13.37%. The pooled
window-5-minus-reservoir difference is -0.0866, while its largest five absolute
pathwise contributions sum to -0.1082. That small ranking is tail-sensitive too.
All 45 method/pair tail diagnostics and indices remain in `two_factor/results.json`.
Saved-model cash replays reproduce these extremes; no accounting discrepancy
explains them, and large errors alone do not identify their statistical cause.

The full process took **116.037 seconds**; internal through-save time was
114.868 seconds and peak RSS 392.86 MiB. Common simulation/summary construction
took 1.198 seconds. Execution used CPU float64 and two BLAS/OpenMP threads on
an Intel Xeon Platinum 8573C. The following component totals cover all candidates
and final draws. Selection is a subset, not an additional cost; inference is the
mean single-draw cost for 8,000 test paths, including features, gains and evaluation.

| Representation | Features (s) | Gain rows (s) | Fitting (s) | Validation (s) | Test evaluation (s) | Full selection (s) | Inference (s/8,000 paths) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current tanh | 8.956 | 0.569 | 0.607 | 0.022 | 0.007 | 5.011 | 0.309 |
| Window 5 | 13.916 | 0.586 | 0.645 | 0.019 | 0.006 | 7.071 | 0.501 |
| Window 20 | 24.519 | 0.549 | 0.607 | 0.020 | 0.008 | 11.934 | 0.868 |
| Two-EWMA tanh | 10.529 | 0.536 | 0.612 | 0.020 | 0.007 | 5.856 | 0.345 |
| Reservoir | 38.640 | 0.690 | 0.633 | 0.019 | 0.005 | 18.883 | 1.328 |

EWMA construction is included in feature time. Random-parameter drawing adds
0.047 seconds in total; remaining process time includes saving and reporting.
These are single-run measurements, not controlled timing estimates. All methods
fit 69 readout coefficients; frozen random-parameter counts are respectively
320/1,600/5,440/448/16,640. The local figure `two_factor/comparison.png` shows
partition-level error differences, selection costs and inference costs. Dots are
partition values; black markers and bars show mean ± partition SD, not confidence
intervals. Generated figures and raw bundles remain excluded from Git.

The three answers are:

1. **History versus current summaries:** EWMA has 5.99% lower pooled MSE, but wins
   only partition 1; current tanh wins partitions 2 and 3. Neither window nor the
   reservoir improves pooled current-summary error. The tail-sensitive EWMA result
   does not establish a consistent benefit from historical representations.
2. **Reservoir versus simple history:** the reservoir beats window 20 in all three
   partition means, but loses to window 5 overall and in two partitions. Window-5
   and reservoir draw ranges overlap in every partition; one window-20/reservoir
   draw comparison also reverses in partition 3. EWMA beats the reservoir in every
   partition and every within-partition comparison between final draws.
3. **Computational value:** the reservoir's extra cost is not justified against
   EWMA here: selection costs 3.22 times as much and inference 3.85 times as much,
   with higher error. EWMA costs 16.86% more selection time and 11.61% more inference
   time than current tanh for its lower pooled MSE. That observed trade-off is
   modest, but its reliability remains unresolved given the partition reversals
   and extreme-path dependence.

**Execution checks:** 84 focused tests passed before declaration. Independent saved-record
aggregation reproduced all 90 candidate means, 15 selections, 45 residual vectors,
15 partition rows, five summaries, 30 paired differences and 45 tail diagnostics:
1,050 numerical checks, maximum discrepancy $4.44\times10^{-16}$. All 22 frozen
source hashes match; three test-price hashes are distinct and disjoint from 31
historical hashes. Replaying 180 representative model/path cases, including each
model's three largest absolute errors, matched saved errors within
$9.81\times10^{-11}$ and nominal-versus-discounted cash accounting within
$1.61\times10^{-13}$. Verification time is outside benchmark timings; no refits
were performed. Previous numerical artifacts, declarations and hashes are unchanged.

**Independent review (29 October 2026): VERIFIED.** The 84 focused tests passed
again. A separate reconstruction verified all 315 fit IDs, 15 validation
selections, errors, paired differences, tail diagnostics and timing totals
(910 numerical comparisons; maximum discrepancy $8.88\times10^{-16}$).
Saved chronology and the runner's selection guard agree: all selections froze
before the first test generation. All 22 frozen source hashes match, and all
269 historical nonreport files match the pre-run inventory. Independently rebuilt
features and direct cash replays for 15 saved strategies on 45 paths, including
each representation/partition's largest absolute error, matched saved errors
within $2.87\times10^{-11}$ and cash accounting within $8.18\times10^{-14}$.
For one saved draw per method in partition 1, all 345 training gain scales matched
exactly; the normalized ridge normal-equation residual was at most $2.50\times10^{-12}$.
No material transition, information, selection or accounting defect was found.
The finite-grid martingale and moment argument is valid; feature timings include
EWMA construction. The figure was visually inspected and its values, units and
descriptive error bars checked. Research code and numerical outputs were unchanged;
no refits or new experiments were run during this review.

Only three partitions, three draws per setting, one fixed grid and a narrow
shortlist were tested. Rankings concern these fitted representations and budgets;
they do not prove an intrinsic memory mechanism. This benchmark observes total
variance and changes the market model, so its errors must not be compared directly
with the older price-only Heston errors as if the information sets were identical.
Artifacts are local under `results/cas_memory/two_factor/`: `plan.json`,
`declaration.json`, `results.json`, three `test_*.npz` archives, summary/paired CSVs,
`timing.json`, `run.log`, the executed-source snapshot and `comparison.png`.

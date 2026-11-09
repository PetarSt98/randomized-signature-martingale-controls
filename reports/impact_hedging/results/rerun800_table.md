| Hedger | features | test net RMSE (mean ± sd over seeds) | liquid regime | illiquid regime | mean cost | CVaR95 | vs. reservoir+tanh (RMSE ratio, 95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static tanh (memoryless) | 260 | **1.678** ± 0.009 | 1.467 | 1.867 | 0.732 | 3.45 | 1.306 [1.297, 1.315] |
| Static tanh + one lag (dS_k, q_{k-1}) | 262 | **1.670** ± 0.010 | 1.454 | 1.864 | 0.737 | 3.45 | 1.300 [1.291, 1.310] |
| EWMA bank, linear readout | 28 | **1.989** ± 0.010 | 1.734 | 2.217 | 0.623 | 4.12 | 1.548 [1.536, 1.559] |
| EWMA bank + tanh head | 284 | **1.425** ± 0.007 | 1.201 | 1.621 | 0.625 | 3.43 | 1.109 [1.104, 1.115] |
| Hand-crafted propagator stats + tanh | 302 | **1.321** ± 0.012 | 0.914 | 1.631 | 0.591 | 3.40 | 1.028 [1.023, 1.032] |
| Reservoir, recurrence removed | 260 | **1.685** ± 0.011 | 1.499 | 1.854 | 0.719 | 3.49 | 1.311 [1.302, 1.321] |
| Reservoir (Petar's readout) | 132 | **1.354** ± 0.011 | 0.970 | 1.653 | 0.593 | 3.54 | 1.053 [1.049, 1.057] |
| Reservoir + tanh head | 260 | **1.285** ± 0.005 | 0.862 | 1.603 | 0.591 | 3.43 | — |
| GRU trained end-to-end (same budget) | 36 | **1.390** ± 0.009 | 1.119 | 1.619 | 0.617 | 3.45 | 1.082 [1.077, 1.087] |
| Oracle (sees F, impact, regime) | 262 | **1.198** ± 0.009 | 0.625 | 1.579 | 0.544 | 3.31 | 0.932 [0.928, 0.937] |

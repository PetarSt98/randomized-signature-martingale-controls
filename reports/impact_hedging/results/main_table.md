| Hedger | features | test net RMSE (mean ± sd over seeds) | liquid regime | illiquid regime | mean cost | CVaR95 | vs. reservoir+tanh (RMSE ratio, 95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static tanh (memoryless) | 260 | **1.679** ± 0.006 | 1.469 | 1.865 | 0.740 | 3.45 | 1.292 [1.284, 1.300] |
| EWMA bank, linear readout | 28 | **1.999** ± 0.010 | 1.745 | 2.224 | 0.622 | 4.13 | 1.538 [1.529, 1.548] |
| EWMA bank + tanh head | 284 | **1.435** ± 0.007 | 1.196 | 1.639 | 0.646 | 3.44 | 1.104 [1.099, 1.109] |
| Hand-crafted propagator stats + tanh | 302 | **1.368** ± 0.013 | 1.028 | 1.639 | 0.628 | 3.38 | 1.053 [1.048, 1.058] |
| Reservoir, recurrence removed | 260 | **1.686** ± 0.004 | 1.488 | 1.862 | 0.737 | 3.47 | 1.297 [1.289, 1.305] |
| Reservoir (Petar's readout) | 132 | **1.362** ± 0.014 | 0.990 | 1.653 | 0.609 | 3.54 | 1.048 [1.044, 1.052] |
| Reservoir + tanh head | 260 | **1.299** ± 0.012 | 0.889 | 1.608 | 0.598 | 3.43 | — |
| GRU trained end-to-end (same budget) | 36 | **1.462** ± 0.011 | 1.218 | 1.671 | 0.635 | 3.49 | 1.125 [1.120, 1.131] |
| Oracle (sees F, impact, regime) | 262 | **1.204** ± 0.012 | 0.639 | 1.579 | 0.545 | 3.31 | 0.927 [0.922, 0.931] |

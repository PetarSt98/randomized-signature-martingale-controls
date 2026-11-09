| hedger | main (seeds [0, 1, 2, 3, 4]) | rerun800 (seeds [5, 6, 7]) | change |
|---|---:|---:|---:|
| Static tanh (memoryless) | 1.679 ± 0.006 | 1.678 ± 0.009 | -0.0% |
| Static tanh + one lag (dS_k, q_{k-1}) | — | 1.670 ± 0.010 | new |
| EWMA bank, linear readout | 1.999 ± 0.010 | 1.989 ± 0.010 | -0.5% |
| EWMA bank + tanh head | 1.435 ± 0.007 | 1.425 ± 0.007 | -0.6% |
| Hand-crafted propagator stats + tanh | 1.368 ± 0.013 | 1.321 ± 0.012 | -3.5% |
| Reservoir, recurrence removed | 1.686 ± 0.004 | 1.685 ± 0.011 | -0.0% |
| Reservoir (Petar's readout) | 1.362 ± 0.014 | 1.354 ± 0.011 | -0.6% |
| Reservoir + tanh head | 1.299 ± 0.012 | 1.285 ± 0.005 | -1.1% |
| GRU trained end-to-end (same budget) | 1.462 ± 0.011 | 1.390 ± 0.009 | -4.9% |
| Oracle (sees F, impact, regime) | 1.204 ± 0.012 | 1.198 ± 0.009 | -0.5% |

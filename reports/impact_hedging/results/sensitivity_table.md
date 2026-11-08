| hidden-regime separation | seeds | static tanh | EWMA+tanh | hand-crafted | reservoir+tanh | oracle | reservoir+tanh vs EWMA+tanh | reservoir+tanh vs static | gap to oracle closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no hidden regime (κ=5 both, λ=0.9) | 1 | 0.839 | 0.820 | 0.832 | 0.823 | 0.817 | -0.4% | +1.9% | n/a (static–oracle gap only 0.022) |
| κ: 0.5 vs 5 | 2 | 0.757 | 0.755 | 0.774 | 0.760 | 0.730 | -0.6% | -0.4% | n/a (static–oracle gap only 0.027) |
| κ: 0.5 vs 10 | 2 | 1.068 | 1.005 | 1.000 | 0.970 | 0.906 | +3.4% | +9.2% | 60% |
| κ: 0.5 vs 20 (main) | 5 | 1.679 | 1.435 | 1.368 | 1.299 | 1.204 | +9.4% | +22.6% | 80% |

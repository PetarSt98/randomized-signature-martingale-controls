### volterra: 10 dataset seeds, 3 fresh draws each, 8000 test paths per seed

| method | pooled test RMSE (mean ± sd over seeds) | RMSE ratio vs static (mean, range) | seeds method beats static | forecast R² of future vol (mean) |
|---|---:|---:|---:|---:|
| affine | 3.152 ± 0.024 | 1.997 [1.766, 2.116] | 0/10 | 0.526 |
| static | 1.584 ± 0.090 | 1 | — | 0.534 |
| window4 | 1.846 ± 0.022 | 1.169 [1.051, 1.235] | 0/10 | 0.583 |
| leaky | 1.679 ± 0.030 | 1.063 [0.987, 1.121] | 2/10 | 0.587 |
| reservoir | 1.826 ± 0.023 | 1.156 [1.047, 1.213] | 0/10 | 0.573 |

### volterra_markov: 10 dataset seeds, 3 fresh draws each, 8000 test paths per seed

| method | pooled test RMSE (mean ± sd over seeds) | RMSE ratio vs static (mean, range) | seeds method beats static | forecast R² of future vol (mean) |
|---|---:|---:|---:|---:|
| affine | 3.116 ± 0.026 | 2.086 [1.880, 2.242] | 0/10 | 0.905 |
| static | 1.497 ± 0.075 | 1 | — | 0.905 |
| window4 | 1.713 ± 0.029 | 1.147 [1.037, 1.216] | 0/10 | 0.905 |
| leaky | 1.625 ± 0.034 | 1.087 [1.010, 1.148] | 0/10 | 0.905 |
| reservoir | 1.690 ± 0.019 | 1.132 [1.033, 1.211] | 0/10 | 0.905 |

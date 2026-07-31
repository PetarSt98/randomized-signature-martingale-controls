# Homogeneous free-Lie rank atlas

## Scope and protocol

Stage A ran all 120 combinations of three architectures, `input_dim` in
`{2, 3}`, `degree` in `{1, 2, 3, 4}`, and `state_dim` in
`{1, 2, 3, 4, 6}`. It used weight seed 1729, point seed 2718, and 18 requested
extra scalar rows. Every matrix had at least twice as many rows as columns.

Stage B concentrated on the smallest scalar-state regimes that Stage A showed
to be either robustly full or tolerance-sensitive. It used weight seeds
`1729, ..., 1733`, point seeds `2718, ..., 2720`, and 15 runs per selected
homogeneous configuration. Each expanded point set contains the exact
Stage-A-protocol point set for its point seed followed by the same number of
additional Gaussian states from local seed `1_000_000_000 + point_seed`.

The per-run classifications used below are:

- `clear_full`: the standard rank and ranks at relative tolerances
  `1e-8`, `1e-10`, and `1e-12` all equal the Witt dimension;
- `clear_deficient`: those four ranks agree below the Witt dimension;
- `tolerance_sensitive`: those four ranks do not agree.

[REPRODUCIBLE NUMERICAL EVIDENCE] All 120 Stage A cases and all 180 Stage B
cases completed. There were no skips, non-finite vector fields, failed SVDs,
or overflows.

[REPRODUCIBLE NUMERICAL EVIDENCE] Stage A required 123.896 seconds wall time.
The corrected Stage B run required approximately 26 minutes 42 seconds wall
time; the sum of recorded per-case times was 1,593.420 seconds.

## 1. Exact architecture formulas

The implemented vector fields are:

1. Principal shallow biased tanh:

   \[
   V_i(x)=\tanh(A_i x+b_i).
   \]

2. Controlled shallow bias-free tanh:

   \[
   V_i(x)=\tanh(A_i x).
   \]

3. Depth-two exponential reference:

   \[
   V_i(x)=\exp\!\left(A_i\exp(D_i x)\right).
   \]

Both exponentials and tanh are componentwise. Each \(D_i\) is structurally
diagonal and is stored by its diagonal entries. The exponential family is a
reference architecture; its activation differs from tanh.

## 2. Exact Lie-bracket convention

The representation uses

\[
[V,W](x)=DW(x)V(x)-DV(x)W(x).
\]

Every matrix column is the represented vector field of one complete
standard-Lyndon basis element in one exact homogeneous degree. One architecture
parameter draw is reused at every evaluation point in a run.

## 3. Homogeneous Witt dimensions

[EXACT FACT FROM ALGEBRA MODULE] The exact homogeneous dimensions are:

| `input_dim` | degree 1 | degree 2 | degree 3 | degree 4 |
|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 2 | 3 |
| 3 | 3 | 3 | 8 | 18 |

[EXACT FACT FROM ALGEBRA MODULE] The atlas columns use every element of these
homogeneous Lyndon bases; no cumulative matrix or single known relation
replaces the complete calculation.

## 4. Stage A homogeneous rank / Witt-dimension table

Each ordinary cell is `standard_rank/Witt_dimension`, with all three relative
ranks equal to the displayed rank. A bracketed triple is shown only when the
`1e-8`, `1e-10`, and `1e-12` ranks differ from the standard rank.

[REPRODUCIBLE NUMERICAL EVIDENCE] Stage A classified 114 cases as
`clear_full` and six as `tolerance_sensitive`.

| Architecture | input | degree | Witt | state 1 | state 2 | state 3 | state 4 | state 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shallow biased tanh | 2 | 1 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| shallow biased tanh | 2 | 2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| shallow biased tanh | 2 | 3 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| shallow biased tanh | 2 | 4 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| shallow biased tanh | 3 | 1 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| shallow biased tanh | 3 | 2 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| shallow biased tanh | 3 | 3 | 8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| shallow biased tanh | 3 | 4 | 18 | 18/18 [16,18,18] | 18/18 | 18/18 | 18/18 | 18/18 |
| shallow bias-free tanh | 2 | 1 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| shallow bias-free tanh | 2 | 2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| shallow bias-free tanh | 2 | 3 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| shallow bias-free tanh | 2 | 4 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| shallow bias-free tanh | 3 | 1 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| shallow bias-free tanh | 3 | 2 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| shallow bias-free tanh | 3 | 3 | 8 | 8/8 [7,8,8] | 8/8 | 8/8 | 8/8 | 8/8 |
| shallow bias-free tanh | 3 | 4 | 18 | 13/18 [8,10,12] | 18/18 | 18/18 | 18/18 | 18/18 |
| depth-two exponential | 2 | 1 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| depth-two exponential | 2 | 2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| depth-two exponential | 2 | 3 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| depth-two exponential | 2 | 4 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| depth-two exponential | 3 | 1 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| depth-two exponential | 3 | 2 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| depth-two exponential | 3 | 3 | 8 | 8/8 [6,7,7] | 8/8 | 8/8 | 8/8 | 8/8 |
| depth-two exponential | 3 | 4 | 18 | 10/18 [6,8,8] | 18/18 [17,18,18] | 18/18 | 18/18 | 18/18 |

## 5. Stage B ranks, singular values, and conditioning

Rank ranges below are ordered as standard / `1e-8` / `1e-10` / `1e-12`.

| Architecture | input | state | degree | points | runs | rank ranges | sigma-min / sigma-max | retained condition |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| shallow biased tanh | 2 | 1 | 1 | 40 | 15 | 2 / 2 / 2 / 2 | 2.87e-2–7.27e-1 | 1.38–34.8 |
| shallow biased tanh | 2 | 1 | 2 | 38 | 15 | 1 / 1 / 1 / 1 | 1.00–1.00 | 1.00–1.00 |
| shallow biased tanh | 2 | 1 | 3 | 40 | 15 | 2 / 2 / 2 / 2 | 2.50e-2–5.75e-1 | 1.74–40.0 |
| shallow biased tanh | 2 | 1 | 4 | 42 | 15 | 3 / 3 / 3 / 3 | 9.62e-4–4.49e-1 | 2.23–1.04e3 |
| shallow biased tanh | 3 | 1 | 3 | 52 | 15 | 8 / 8 / 8 / 8 | 2.58e-7–1.18e-3 | 8.49e2–3.88e6 |
| shallow biased tanh | 3 | 1 | 4 | 72 | 15 | 18 / 15–18 / 16–18 / 17–18 | 9.27e-14–1.92e-7 | 5.22e6–1.08e13 |
| shallow bias-free tanh | 2 | 1 | 4 | 42 | 15 | 3 / 3 / 3 / 3 | 6.98e-6–3.70e-2 | 27.0–1.43e5 |
| shallow bias-free tanh | 3 | 1 | 3 | 52 | 15 | 7–8 / 5–8 / 6–8 / 7–8 | 1.86e-15–3.69e-7 | 2.71e6–3.63e11 |
| shallow bias-free tanh | 3 | 1 | 4 | 72 | 15 | 11–15 / 7–11 / 8–13 / 10–14 | 4.81e-22–4.21e-17 | 2.77e11–6.01e13 |
| depth-two exponential | 2 | 1 | 4 | 42 | 15 | 3 / 3 / 3 / 3 | 2.54e-7–2.26e-5 | 4.43e4–3.94e6 |
| depth-two exponential | 3 | 1 | 3 | 52 | 15 | 6–8 / 4–6 / 5–7 / 5–7 | 3.05e-17–7.72e-13 | 1.21e12–2.59e13 |
| depth-two exponential | 3 | 1 | 4 | 72 | 15 | 7–10 / 4–6 / 5–8 / 6–9 | 9.00e-19–8.32e-18 | 6.53e11–5.31e13 |

[REPRODUCIBLE NUMERICAL EVIDENCE] The replicated binary shallow-biased
configuration is full through every degree from 1 to 4 for all 15 seed pairs
and every configured tolerance.

[NUMERICAL AMBIGUITY] The ternary shallow-biased degree-four matrices are
standard-rank full in all 15 runs, but two of five weight draws lose rank at
one or more relative tolerances. Their retained condition numbers reach
approximately \(1.08\times10^{13}\).

## 6. Stability across point seeds

The table reports how many of the five weight draws gave the same four-rank
tuple across all three point seeds.

| Architecture and homogeneous configuration | point-stable weights | weights full at all tolerances |
|---|---:|---:|
| biased tanh, input 2, state 1, degree 1 | 5/5 | 5/5 |
| biased tanh, input 2, state 1, degree 2 | 5/5 | 5/5 |
| biased tanh, input 2, state 1, degree 3 | 5/5 | 5/5 |
| biased tanh, input 2, state 1, degree 4 | 5/5 | 5/5 |
| biased tanh, input 3, state 1, degree 3 | 5/5 | 5/5 |
| biased tanh, input 3, state 1, degree 4 | 5/5 | 3/5 |
| bias-free tanh, input 2, state 1, degree 4 | 5/5 | 5/5 |
| bias-free tanh, input 3, state 1, degree 3 | 4/5 | 3/5 |
| bias-free tanh, input 3, state 1, degree 4 | 3/5 | 0/5 |
| exponential reference, input 2, state 1, degree 4 | 5/5 | 5/5 |
| exponential reference, input 3, state 1, degree 3 | 5/5 | 0/5 |
| exponential reference, input 3, state 1, degree 4 | 1/5 | 0/5 |

[REPRODUCIBLE NUMERICAL EVIDENCE] Point-seed changes do not affect any rank
tuple in the selected binary shallow-biased configuration.

[NUMERICAL AMBIGUITY] Point-seed instability is concentrated in the most
ill-conditioned ternary scalar cases. It is therefore evidence about
numerical rank, not an exact functional identity.

## 7. Stability across weight seeds

[REPRODUCIBLE NUMERICAL EVIDENCE] Every weight draw in the selected binary
shallow-biased configuration is full through degree 4 at all tolerances.

[REPRODUCIBLE NUMERICAL EVIDENCE] At ternary degree 4, all five shallow-biased
weight draws are standard-rank `18/18`; three are full at every relative
tolerance, while the other two have point-stable rank tuples
`(18,17,18,18)` and `(18,15,16,17)`.

[NUMERICAL AMBIGUITY] The ternary degree-four shallow-biased conditioning is
strongly weight-dependent: the singular-value ratio spans
`9.27e-14` to `1.92e-7`.

## 8. First degree with a reproducible deficiency

[REPRODUCIBLE NUMERICAL EVIDENCE] The earliest all-weight deficiency at any
configured relative threshold occurs at degree 3 in the depth-two exponential
reference: all 15 `1e-8`, `1e-10`, and `1e-12` ranks are below 8. Standard
ranks range from 6 to 8, so this remains tolerance-sensitive.

[REPRODUCIBLE NUMERICAL EVIDENCE] The first replicated configuration that is
deficient in every run at the standard tolerance and at all three relative
tolerances is shallow bias-free tanh with `input_dim=3`, `state_dim=1`,
`degree=4`. Its standard ranks range from 11 to 15 of 18.

[REPRODUCIBLE NUMERICAL EVIDENCE] No replicated shallow-biased configuration
is deficient for every weight and point seed. The binary target is fully
ranked; the ternary degree-four shortfall is tolerance- and weight-dependent.

[NUMERICAL AMBIGUITY] Reproducible deficiency at a specified numerical
tolerance does not establish a vanishing Lie polynomial.

## 9. Biased versus bias-free shallow tanh

Equal seeds give the biased and bias-free systems identical weight matrices;
only the biased system receives its additional sampled bias vectors.

[REPRODUCIBLE NUMERICAL EVIDENCE] For binary scalar degree 4, both systems are
full in all 15 runs. The biased singular-value ratio is larger in every paired
run, by a factor ranging from 4.95 to \(5.30\times10^4\), with a median factor
of 15.4.

[REPRODUCIBLE NUMERICAL EVIDENCE] For ternary scalar degree 4, biased tanh is
standard-rank 18 in all runs, whereas bias-free standard ranks range from 11
to 15. The paired biased standard-rank advantage is 3 to 7 columns; at
`1e-8` it is 7 to 9 columns.

[REPRODUCIBLE NUMERICAL EVIDENCE] The ternary degree-four biased
singular-value ratios exceed their paired bias-free values by factors from
\(1.93\times10^8\) to \(1.18\times10^{11}\).

[UNPROVED CONJECTURE] Bias may remove severe scalar-state degeneracies of the
bias-free family. The atlas does not prove this as an architecture theorem.

## 10. Depth-two exponential reference

[REPRODUCIBLE NUMERICAL EVIDENCE] The binary scalar degree-four exponential
reference is full in all 15 runs, but its condition numbers
(`4.43e4` to `3.94e6`) are much worse than the selected biased target.

[REPRODUCIBLE NUMERICAL EVIDENCE] Its ternary scalar degree-three and
degree-four ranks are deficient at every configured relative tolerance, with
standard ranks varying across weight draws.

[NUMERICAL AMBIGUITY] These comparisons do not isolate depth. The exponential
reference differs from shallow tanh in activation as well as depth.

## 11. Rank changes with tolerance

[REPRODUCIBLE NUMERICAL EVIDENCE] Stage A has six tolerance-sensitive cases:

- biased tanh: input 3, state 1, degree 4;
- bias-free tanh: input 3, state 1, degrees 3 and 4;
- exponential reference: input 3, state 1, degrees 3 and 4;
- exponential reference: input 3, state 2, degree 4.

[REPRODUCIBLE NUMERICAL EVIDENCE] Across the combined atlas, 63 of 300 runs
are tolerance-sensitive: six in Stage A and 57 in the concentrated Stage B
replications. The other 237 are clear-full; no run meets the stricter
`clear_deficient` definition because deficient rank counts change with
tolerance.

[NUMERICAL AMBIGUITY] A tolerance-dependent rank is treated only as evidence
of ill-conditioning. It is not treated as an exact nullity.

## 12. Effect of additional evaluation points

Twelve Stage B cases exactly extend their matching Stage A point set for
weight seed 1729 and point seed 2718. All strict prefixes were verified.

[REPRODUCIBLE NUMERICAL EVIDENCE] Eleven of the 12 rank tuples are unchanged.
For shallow biased tanh with `input_dim=3`, `state_dim=1`, `degree=4`,
doubling points from 36 to 72 changes the tuple from
`(18,16,18,18)` to `(18,17,18,18)`.

[REPRODUCIBLE NUMERICAL EVIDENCE] The extra rows remove one apparent
`1e-8` null direction in that draw, but they do not make it clear-full.

[NUMERICAL AMBIGUITY] No case changes from a reproducibly deficient pattern to
a stable full-rank pattern solely by adding points.

## 13. Tanh saturation and exponential overflow

For tanh, the recorded numerical near-saturation thresholds are derivative
factors at most `1e-8`, `1e-10`, `1e-12`, machine epsilon, and exactly zero.

[REPRODUCIBLE NUMERICAL EVIDENCE] No Stage A or Stage B tanh run has any
derivative factor at most `1e-8`, and none has an exactly zero float64
derivative factor.

[REPRODUCIBLE NUMERICAL EVIDENCE] The largest Stage B tanh preactivation
magnitude is 7.002, with minimum derivative factor `3.31e-6`; that run is
clear-full. No observed rank loss can therefore be attributed to the recorded
near-saturation threshold.

[REPRODUCIBLE NUMERICAL EVIDENCE] No exponential run overflows. The smallest
recorded distance from an inner or outer exponent argument to the float64 log
maximum is approximately 709.27 in Stage A and 709.40 in Stage B.

## 14. Candidate numerical nullspaces

To test whether draw-specific nullspaces contain a common Lyndon coefficient
direction, every Stage B matrix in a configuration was divided by its spectral
norm and the 15 matrices were stacked vertically.

| Architecture | input | degree | stacked shape | sigma-min / sigma-max | stacked ranks std / 1e-8 / 1e-10 / 1e-12 |
|---|---:|---:|---:|---:|---|
| biased tanh | 3 | 4 | 1080x18 | 1.82e-2 | 18 / 18 / 18 / 18 |
| bias-free tanh | 3 | 3 | 780x8 | 2.84e-2 | 8 / 8 / 8 / 8 |
| bias-free tanh | 3 | 4 | 1080x18 | 2.27e-4 | 18 / 18 / 18 / 18 |
| exponential reference | 3 | 3 | 780x8 | 3.10e-5 | 8 / 8 / 8 / 8 |
| exponential reference | 3 | 4 | 1080x18 | 9.70e-9 | 18 / 17 / 18 / 18 |

[REPRODUCIBLE NUMERICAL EVIDENCE] The shallow biased and bias-free stacked
matrices have no common numerical nullspace at any configured tolerance.
In particular, the substantial individual bias-free degree-four nullspaces
rotate with the parameter draw rather than producing one stable coefficient
direction.

[NUMERICAL AMBIGUITY] The exponential degree-four stack has one `1e-8`
candidate null direction, but it disappears at the standard, `1e-10`, and
`1e-12` tolerances. Its maximum normalized per-run residual is approximately
`2.08e-8`.

[NUMERICAL AMBIGUITY] None of these floating nullspaces is an exact Lie
identity.

## 15. Candidate well-conditioned full-rank minor

The selected target is the best-conditioned degree-four run among the 15
replications of the grid-smallest shallow-biased configuration that is full
through degree 4:

- architecture: `shallow_biased_tanh`;
- `input_dim=2`;
- `state_dim=1`;
- through degree 4, with degree 4 as the certificate target;
- weight seed 1729;
- point seed 2719;
- extension point seed 1,000,002,719;
- 42 evaluation points;
- complete degree-four Lyndon column count 3.

Its scalar parameters are

\[
A_1=-1.1257398491596682,\quad
A_2=-0.005658299322056218,
\]

\[
b_1=-1.3974639278267806,\quad
b_2=1.4364079096034785.
\]

The full `42x3` matrix has singular values

`[5.8779461159, 4.8303164269, 2.6408820760]`

and condition number 2.226.

Pivoted QR of the matrix transpose selected scalar rows `[18, 32, 22]`,
corresponding to states approximately
`[-1.1899977347, -2.1409444480, -0.3280588940]`. The resulting complete-basis
`3x3` candidate minor has:

- singular values `[2.3822113319, 1.7642356139, 1.1996348876]`;
- condition number 1.986;
- floating determinant sign `-1`;
- floating log absolute determinant `1.6177639541`.

Its columns are all three degree-four Lyndon brackets:

1. `[X1,[X1,[X1,X2]]]`;
2. `[X1,[[X1,X2],X2]]`;
3. `[[[X1,X2],X2],X2]`.

[REPRODUCIBLE NUMERICAL EVIDENCE] This is a well-conditioned numerical
candidate minor using the complete homogeneous basis.

[NUMERICAL AMBIGUITY] Its floating determinant is not an exact certificate.

## 16. Conclusion ledger and selected branch

[EXACT FACT FROM ALGEBRA MODULE] The tested complete homogeneous basis sizes
are exactly the Witt dimensions displayed in Section 3.

[REPRODUCIBLE NUMERICAL EVIDENCE] The shallow biased tanh architecture with
`input_dim=2`, `state_dim=1` is full through degree 4 in all 15 expanded
replications and all configured tolerance regimes.

[REPRODUCIBLE NUMERICAL EVIDENCE] No shallow-biased deficiency is stable
across all replicated weight draws, point draws, and tolerances. Cross-weight
stacking yields no common shallow-biased numerical nullspace.

[NUMERICAL AMBIGUITY] The ternary scalar shallow-biased degree-four system is
standard-rank full but too ill-conditioned at some weight draws to serve as
the first exact-certificate target.

[UNPROVED CONJECTURE] An exact symbolic or rational evaluation of the selected
minor may establish a nonzero Hall/Lyndon minor for this finite configuration.
That calculation has not been performed here.

**Selected next branch: A.**

The exact-certificate target is:

```text
architecture = shallow_biased_tanh
input_dim = 2
state_dim = 1
through_degree = 4
target_degree = 4
weight_seed = 1729
point_seed = 2719
extension_point_seed = 1000002719
point_count = 42
```

This is the smallest target in the executed grid, not a claim of mathematical
minimality. Branch A means only that the next task should seek a rigorously
nonzero exact evaluation minor. It does not claim generic faithfulness.

## Five most important findings

1. [REPRODUCIBLE NUMERICAL EVIDENCE] All 120 Stage A cases were feasible and
   finite; 114 were clear-full and six were tolerance-sensitive.
2. [REPRODUCIBLE NUMERICAL EVIDENCE] Binary scalar shallow-biased tanh is full
   through degree 4 in every one of 15 replicated seed pairs.
3. [NUMERICAL AMBIGUITY] Ternary scalar shallow-biased degree 4 is
   standard-rank full but weight-dependent at relative tolerances.
4. [REPRODUCIBLE NUMERICAL EVIDENCE] Bias-free ternary scalar degree 4 is
   deficient in every run, but its cross-weight stacked matrix is full, so
   no common Lie-polynomial coefficient direction is identified.
5. [REPRODUCIBLE NUMERICAL EVIDENCE] The selected complete-basis candidate
   minor is `3x3` with condition number 1.986, making Branch A the focused
   next exact step.

## Saved artifacts

- Combined tidy CSV: `research_output/rank_atlas.csv`
- Combined JSON with spectra, parameters, points, nullspaces, and aggregates:
  `research_output/rank_atlas.json`
- Selected target manifest:
  `research_output/selected_target/manifest.json`
- Parameters: `research_output/selected_target/parameters.npz`
- Evaluation points: `research_output/selected_target/evaluation_points.npy`
- Evaluation matrix: `research_output/selected_target/evaluation_matrix.npy`
- Singular spectrum: `research_output/selected_target/singular_values.npy`
- Candidate minor arrays:
  `research_output/selected_target/candidate_minor.npz`
- Candidate minor metadata:
  `research_output/selected_target/candidate_minor.json`


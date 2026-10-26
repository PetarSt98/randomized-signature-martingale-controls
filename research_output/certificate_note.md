# Exact homogeneous degree-one-to-four certificates

## Exact witness

Status: `exact_nonzero` for every homogeneous degree `m=1,2,3,4`.

For `shallow_biased_tanh`, `input_dim=2`, and `state_dim=1`, set

```text
q = atanh(1/2) = log(3)/2
(a1, b1, a2, b2) = (1, 0, 0, q)
V1(x) = tanh(x)
V2(x) = tanh(q) = 1/2
```

Exactly, `exp(2q)=3`, so `tanh(q)=1/2` and `tanh(-q)=-1/2`.
The repository's standard-Lyndon order is checked before each exact matrix is
constructed. Using `[V,W]=W'V-V'W`, the exact `Fraction` activation-jet
recursion gives:

### Degree 1

Evaluation points: `(0, log(3)/2)`.

Basis:

1. `X1`
2. `X2`

```text
[0  1/2]
[1/2  1/2]
```

Exact determinant: `-1/4 != 0`.

### Degree 2

Evaluation points: `(0)`.

Basis:

1. `[X1,X2]`

```text
[-1/2]
```

Exact determinant: `-1/2 != 0`.

### Degree 3

Evaluation points: `(-log(3)/2, log(3)/2)`.

Basis:

1. `[X1,[X1,X2]]`
2. `[[X1,X2],X2]`

```text
[15/32  3/16]
[15/32  -3/16]
```

Exact determinant: `-45/256 != 0`.

### Degree 4

Evaluation points: `(-log(3)/2, 0, log(3)/2)`.

Basis:

1. `[X1,[X1,[X1,X2]]]`
2. `[X1,[[X1,X2],X2]]`
3. `[[[X1,X2],X2],X2]`

```text
[-57/128  -3/32  3/64]
[-1/2  0  1/4]
[-57/128  3/32  3/64]
```

Exact determinant: `135/8192 != 0`.

## Combined certificate

The exact determinant product is

```text
D1 * D2 * D3 * D4 = -6075/16777216 != 0.
```

Therefore the representation is injective on each homogeneous free-Lie
component `L_{2,m}`, `m=1,2,3,4`.

Lemma: If evaluations of vector fields `F1,...,Fr` at finitely many points
form a full-column-rank matrix, then `F1,...,Fr` are linearly independent as
functions. Indeed, a functional linear relation evaluates to a vector in the
matrix nullspace, which is trivial at full column rank. Applying this to the
images of each complete Lyndon basis proves the stated homogeneous
injectivity.

## Analytic genericity argument

For the fixed degree-specific point sets above, each selected determinant
`D_m(a1,b1,a2,b2)` is real analytic on connected `R^4`. Their product
`P=D1*D2*D3*D4` is real analytic and the exact witness gives
`P(1,0,0,q)=-6075/16777216 != 0`, so `P` is not identically zero. Its zero set has
four-dimensional Lebesgue measure zero. Thus all four homogeneous components
are injective simultaneously almost surely under any parameter law absolutely
continuous with respect to four-dimensional Lebesgue measure.

This probability statement excludes singular or fixed-bias laws and the
literal finite-support distribution of a finite-precision seeded PRNG.

## Numerical checks and preserved atlas target

The reproduction program cross-checks every exact matrix through the PyTorch
autodiff pipeline. All exact-to-binary64 conversions are explicitly reported;
these checks are non-certifying numerical evidence only.

The original seed-1729 `42 x 3` degree-four atlas target, including pivoted-QR
rows `[18,32,22]`, is preserved separately as `numerical_candidate_only`.

## Scope limitation

This proves simultaneous degree-by-degree homogeneous injectivity for degrees
one through four. It does not prove ungraded cross-degree independence or
injectivity on the ungraded direct sum, because images from different degrees
could still cancel.

The separate [sharp fixed-state Hall-rank theorem](intrinsic_hall_rank_theorem.md)
proves generic rank min(D, nu) in the full biased-tanh parameter space and
local recovery of the truncated Hall response from one zero-initialized vector
at D >= nu. It uses a different witness. This scalar certificate and its
cross-degree scope remain unchanged; none of its constants transfers to that
construction.

Reproduce with:

```text
python research_output/reproduce_certificate.py
```

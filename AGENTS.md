# Research repository instructions

This repository contains quantitative-finance experiments and a separate
`lie_reservoir` mathematical research module.

## Current research direction

The active research question is:

> When does a shallow componentwise-tanh randomized-signature reservoir preserve
> enough intrinsic free-Lie information to identify a truncated log-signature?

The existing exact theorem package

    d = 2
    D = 1
    L <= 4

is a certified base case and must remain unchanged unless explicitly requested.

The current extension programme is narrowly fixed to:

1. general graded identifiability conditional on homogeneous free-Lie injectivity,
   for arbitrary finite `d`, `D`, and `L`;

2. lifting the existing exact `d=2, D=1, L<=4` witness to arbitrary
   state dimension `D>=1`;

3. auditing and, only if valid, proving a general finite-width faithfulness
   theorem for shallow componentwise-tanh vector fields;

4. for the general-width proof, first establish a linear-to-tanh transfer
   principle and then determine the strongest independence statement actually
   implied by the same proof:
   - homogeneous Lie faithfulness;
   - joint Hall/free-Lie faithfulness across degrees;
   - or word-level iterated-field independence.
5. auditing and, only if valid, proving a sharp fixed-state Hall-rank theorem
   for the degree-L truncated response:

       rank H_theta = min(D, nu_{d,L})

   generically in the full biased shallow-tanh parameter space, where
   nu_{d,L} is the dimension of the truncated free-Lie space.

   The intended consequence is dimension-optimal local recovery of the
   truncated Hall coordinates from one zero-initialized D-dimensional
   terminal vector when D >= nu_{d,L}.

   This is a candidate theorem, not an established result.

Do not introduce another research direction unless explicitly instructed.

## Research discipline

- Prefer one rigorous structural theorem over many special cases.
- Do not assume a requested theorem is true.
- If a theorem is false or needs an additional assumption, stop and report the
  smallest precise obstruction.
- Numerical evidence is not proof.
- Floating-point rank or determinants are not exact certificates.
- Keep the following observation objects strictly distinct:
  - ideal graded scale jets;
  - degree-L truncated finite-scale responses;
  - actual full CDE responses.
- Do not promote homogeneous injectivity to ungraded injectivity without proof.
- Do not promote truncated reconstruction to full-CDE reconstruction.
- Do not claim novelty from classical free-Lie, PBW, Ado, faithful
  representation, signature, or Chen--Fliess theory.
- Clearly separate:
  - proved theorem;
  - proof attempt;
  - computational check;
  - numerical evidence;
  - conjecture.
- Do not claim results beyond the exact proved scope.

## Existing certified results

Treat the existing `d=2, D=1, L<=4` package as frozen and authoritative.

It currently includes:

- exact homogeneous free-Lie injectivity through degree four;
- exact shallow-tanh witness and rational determinant certificates;
- exact cross-degree dependence at the witness;
- graded Hall/log-signature reconstruction from ideal scale jets;
- explicit eight-output degree-four truncated observation design;
- fixed-design genericity;
- local analytic inversion;
- exact local inverse-Lipschitz stability.

Do not weaken, rewrite, or silently generalize these statements.

## Allowed current work

The following are currently in scope:

### A. General graded identifiability

Generalize the conditional reconstruction framework to arbitrary finite
`d`, `D`, and `L`, assuming

    rho_m : L_{d,m} -> X(R^D)

is injective for every `m <= L`.

This may include:

- arbitrary-degree tensor/log-signature triangularity;
- vector-valued scale jets;
- finite point-coordinate evaluation lemmas;
- joint state-scale-coordinate independence;
- finite truncated observation maps;
- local inverse statements for the truncated model;
- precise converse/kernel statements.

### B. State-dimension lifting

Prove and verify the embedding of the existing scalar witness into every
`D >= 1` for

    d = 2
    L <= 4.

Treat this as a corollary, not as a width-expressivity theorem.

### C. General shallow-tanh faithfulness

A proof attempt may investigate a theorem of the form

    D >= D0(d,L)
        => generic finite-degree faithfulness

for shallow componentwise-tanh vector fields.

The preferred structural route is:

    faithful finite-dimensional linear vector fields
    -> small-epsilon componentwise-tanh perturbation
    -> nonzero leading determinant
    -> analytic genericity.

For the explicit tensor construction, use the repository-compatible
right-concatenation convention only after it has been verified against the
existing word/operator convention.

The candidate bound

    D0(d,L) = 1 + d + ... + d^L

is a proposed sufficient bound, not an established theorem and not a
minimal-width claim.

Before restricting a successful proof to homogeneous Lie injectivity, check
whether exactly the same argument proves the stronger:

- joint Hall independence across degrees; or
- word-level iterated-field independence through level `L`.

State only the strongest result actually proved.

### D. Sharp fixed-state Hall rank

A proof attempt may investigate the candidate theorem

    rank H_theta = min(D, nu_{d,L})

for the matrix of Hall-bracket evaluations at one fixed initial state,
preferably zero.

This work may include:

- an explicit grading-derived affine witness;
- transfer of that witness to actual componentwise-tanh vector fields;
- exact generic-rank certificates;
- the cases D < nu, D = nu, and D > nu;
- local inversion of the degree-L truncated Hall response;
- the matching dimensional lower bound for one-vector local recovery.

Keep this distinct from:

- word-response independence;
- function-level joint field independence;
- actual full-CDE reconstruction;
- linear-readout approximation;
- practical conditioning of random reservoirs.

Do not assume the candidate theorem is true. If the proof fails or the result
is already covered by prior work, stop and report the precise obstruction.

## Frozen / out-of-scope directions

Do not work on the following unless explicitly requested:

- scalar degree-5-to-8 frontier or Lie identities;
- `d=8, D=256` or other production-sized rank case studies;
- broad `(d,D,L)` numerical atlases;
- width optimization or minimal faithful dimension beyond the explicitly
  authorized fixed-state Hall-rank / truncated local-recovery problem;
- architecture comparisons;
- full-CDE `B -> C` remainder reconstruction;
- rough-path extensions;
- discrete-time reconstruction;
- finance, Heston, pricing, control-variate, or hedging extensions;
- new empirical experiments.

If the general-width route fails or is already covered by prior work, stop and
report that. Do not compensate by starting unrelated numerical experiments.

## Code discipline

- Search the repository before creating code.
- Reuse existing word, Lyndon/Hall, bracket, activation-jet, exact-matrix, and
  certificate implementations.
- Do not create a second free-Lie implementation.
- Do not create a second tensor-word or bracket convention.
- Keep new code minimal, explicit, deterministic, and mathematically auditable.
- Exact proof paths must use exact arithmetic where applicable.
- Code may verify a mathematical proof but must not substitute for one.
- Avoid generic frameworks, registries, factories, service layers, and
  unnecessary abstraction.
- Avoid redundant code and AI-generated boilerplate.
- Do not modify existing certified theorem artifacts unless explicitly required.

## Mathematical convention discipline

Before using a new tensor or matrix representation, verify compatibility with
the repository conventions for:

- chronological word order;
- operator-composition order;
- Hall/Lyndon ordering;
- bracket sign;
- tensor concatenation;
- Taylor/factorial normalization.

A convention mismatch is a blocking issue, not something to patch with an
unexplained sign or reversal.

## Validation

Before accepting code changes run:

    python -m pytest -q
    git diff --check

Do not delete existing research artifacts or certificate files unless explicitly
instructed.
# Research repository instructions

This repository contains quantitative-finance experiments and a separate
`lie_reservoir` mathematical research module.

## Current research direction

The active research direction is fixed:

homogeneous Lie injectivity
-> exact graded log-signature reconstruction
-> eight finite state-scale observations
-> explicit local stability.

Do not broaden this direction unless explicitly instructed.

## Research discipline

- Prefer one small rigorously proved result over multiple speculative results.
- If a requested theorem is false or needs an additional assumption, stop and
  report the precise obstruction.
- Numerical evidence is not proof.
- Floating-point rank or determinants are not exact certificates.
- Keep ideal scale jets, truncated finite-scale responses, and full CDE
  responses distinct.
- Do not claim results beyond the exact proved scope.

## Code discipline

- Search before creating new code.
- Reuse existing implementations before extending them.
- Do not duplicate free-Lie, Lyndon-basis, bracket, activation-jet, or
  certificate logic.
- Keep code small, explicit, and mathematically auditable.
- No generic frameworks, registries, factories, service layers, or unnecessary
  abstraction.
- Exact proof paths use exact arithmetic where applicable.
- Tests must be deterministic.
- Follow SOLID principles and clean code
- No redundant code and AI slop

## Scope

Do not modify finance, Heston, pricing, control-variate, or hedging code unless
explicitly requested.

Do not extend to:
- d=3;
- degree 5;
- larger reservoir state dimension;
- architecture comparisons;
- finance experiments;
- discrete-time reconstruction;

unless explicitly requested.

## Validation

Before accepting changes run:

    python -m pytest -q
    git diff --check

Do not delete existing research artifacts or certificate files unless explicitly
instructed.
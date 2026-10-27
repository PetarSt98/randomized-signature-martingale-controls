# CAS independent-review handoff

Review branch: `cas/completed-studies-review-2026-10-27`.
Base: `a87c34cc4d55d2680c4aad22c8870d629003ddbc`.
The current handoff commits the completed research without changing its results,
rerunning experiments, or replacing historical filtering experiments.

## Contents

- [Frictionless study](cas_frictionless/REPORT.md): declaration, 44 fit records,
  summary, residual arrays, Transformer checkpoint, figure, logs and validation.
- [Cost-aware study](cas_costs/REPORT.md): declaration, 220 fit records, 36 evaluation
  archives, CSV summary, checkpoints, figure, logs and validation.
- [Transfer and reliability study](cas_transfer/REPORT.md): declaration, 360 fit
  records, 132 unique evaluation archives, per-run and aggregated CSVs, checkpoints,
  three figures, environment metadata, independent audits and validation.
- [Methods and findings](../README.md), implementation in `rsmc/`, four CAS scripts
  in `scripts/`, and five CAS test files in `tests/`.
- [Evidence manifest](cas_evidence_manifest.json): sizes and SHA256 checksums for
  all 228 pre-existing study artifacts (83,723,833 bytes). The manifest does not
  include itself or this handoff note. All listed files are intended for Git.

The saved suite result is 91 passing tests. Studies have zero fit failures;
optimizer budget exhaustion is recorded separately and is not convergence.
The handoff operation performed inventory/integrity checks only, not experiments.

## Binary evidence and access

No required study evidence is intentionally excluded from this commit. It includes
168 NPZ cash-flow/error archives, eight NPY residual arrays, 15 PyTorch checkpoints,
and five PNG figures. Python/pytest caches are ignored; study evidence is not.
Largest artifact: `cas_transfer/results.json`, 8,768,481 bytes. No individual file
requires exclusion under normal GitHub file-size limits.

Publication of the branch is required before a new cloud checkout can access it.
A local commit alone is not a remote handoff. If the browser pull-request action
omits binary files, the new task will lack pathwise cash flows/residuals,
Transformer checkpoints and potentially figures despite seeing reports and CSVs.
The local originals are retained unchanged; the manifest identifies every file
needed to detect such an incomplete publication. The final chat response records
whether a direct push succeeded or which publication step remains.

Fresh review tasks must select this branch (or its published PR head). `main`
is not modified. The prepared virtual environment and workspace setup helpers
are outside the repository; use `requirements.txt` and the recorded environment
versions when creating a new environment. They are runtime setup, not lost
empirical evidence. Raw simulator path tensors and per-date trade tensors were
not saved by the studies; saved seeds/source fingerprints and tested accounting
are the existing reproducibility basis, not an artifact removed by this handoff.

## Integrity and independent review

Check every manifest path against its recorded byte count and SHA256 after
checkout. Saved `results.json` files contain configurations, premiums, seeds,
source fingerprints, statuses, times and selected fits. Existing declarations
must remain unchanged. The pre-transfer cost module is preserved at
`cas_transfer/provenance/cost_hedging_reference.py`, with the exact hash from the
cost study; this explains the intentional later multiasset extension.

The saved independent transfer audit is reusable without fitting models:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/.venvs/rsmc/bin/python results/cas_transfer/provenance/independent_audit.py
```

Use the Python executable for the review environment if that virtual-environment
path differs. The audit checks saved results and regenerates simulator path
fingerprints, not model fits. Its completed output is already included.

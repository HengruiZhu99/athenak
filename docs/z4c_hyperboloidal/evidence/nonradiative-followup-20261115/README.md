# Wider-pulse evidence

Use the [follow-up report](../../nonradiative-followup-20261115/report.md) for
interpretation and [reproduction instructions](../../nonradiative-followup-20261115/reproduce.md)
for commands, immutable snapshots and raw-data locations. The campaign and
its explicit analysis recovery are complete; the accuracy verdict remains qualified.

- `measurements.json`, plot PNG/PDF files, and `*-common-cuts.npz`: all-mode
  absolute errors on common Bondi cuts, full native tensor errors, cut transport,
  constraints, timestep/frame controls and the isolated reference comparison.
- `pulse-resolution.json`, `characteristics.*`: radial-only resolution survey
  and frozen outgoing-characteristic duration estimate. Neither certifies the
  full native operator or that the background has settled.
- `initial-data-check.json`, `boundary-admission.json`: unchanged physical seed
  and actual endpoint admission for the wider pulse.
- `reference-regression.json`: original-header failure and corrected-header
  success for independent stationary-vacuum identities. Finite-value checks
  supplement the numerical thresholds.
- `reference-guard-regression.json`: a non-flat-Omega counterexample preventing
  analytic shift jets from bypassing the inner gauge. The final guard is
  redundant for all campaign configurations; the immutable GPU candidate
  predates it. `provenance/reference-guard-check.json` records bitwise equality
  of guard-only initial/two-step CPU evolutions.
- `reference-accuracy-paired.csv`: exploratory six-column radial inventory,
  with normalized line endings, scope and raw-file hashes in `provenance/reference-inventory.json`. The
  separately reproducible regression is the acceptance evidence.
- `ctest.log`: all 15 focused tests after the final full rebuild.
- `provenance/checkpoint-audit.json`: all eight final checkpoint hashes,
  endpoints and finite-field checks.
- `provenance/halfdt-9094227/analysis-recovery.json`: analysis-only recovery
  with unchanged checkpoint hash. The original exit 143 and import-failure
  log are preserved alongside the successful recovery scheduler receipt.
  `alignment-refactor-check.json` records bitwise-identical comparison outputs
  after removing plotting dependencies.
- `legacy-N12-peak-attribution.json`: signed, additive instantaneous tensor
  attribution for the old narrow pulse. It is not an evolved ablation.
- `native-summaries/`: principal and corrected-control summaries, signed
  witnesses, compact numerical series, all retained modes and full sampled
  history rows. History JSON retains original text hashes and column headers.
  Separate cadence-three and frame-cutoff mode archives permit independent
  checks of the temporal/frame comparisons. Series/mode NPZ files are lossless selections from the larger raw products;
  the source hashes are recorded in the final integrity check.
- `inputs/`: exact inputs, named with their run to distinguish principal,
  corrected-reference and timestep controls.
- `provenance/`: source/build/runtime/launcher receipts, seed comparisons,
  scheduler completion/costs, snapshot and package hashes. Text receipts have
  trailing whitespace trimmed; original bytes remain in the raw result root.
  Run-specific source manifests must not be replaced by the final source
  manifest. Report/helper updates do not retrospectively redefine a run.
- `artifact-sha256.json`: hashes of compact evidence, excluding itself.

Large native records, metric caches, pointwise tensors and checkpoints remain
at `/lus/flare/projects/CompactBinaryMerger/hzhu/nonradiative-followup-20261115`.
Collected products and local checks are at
`/Users/hz0693/research/results/nonradiative-followup-20261115`.
The earlier campaign's evidence remains unchanged.

# Compact evidence index

The authoritative interpretation is [the report](../../nonradiative-20261115/report.md),
with diagnostic conventions in [method.md](../../nonradiative-20261115/method.md).

- `measurements.json`: common-Bondi-cut mode comparisons, native regularity/frame
  summaries, analytic case summaries, cadence and half-step comparisons.
- `native-*.png/.pdf`, `analytic-errors.png/.pdf`: exportable plots. The tensor
  path-difference plot omits the three one-sided edge stencils at each end;
  original series retain them. Native maxima use the same interior convention.
- `control-common-cuts.npz`, `gauge-common-cuts.npz`: every retained complex mode
  on common physical cuts, with resolution and mode labels.
- `provenance/`: source/build/runtime/launcher receipts and scheduler records.
  The final committed source manifest is separate from the archived run-specific
  snapshots. Module/scheduler text and compact initial-history copies have trailing whitespace trimmed;
  original logs remain in the raw roots. Run receipts hash the original histories;
  the artifact manifest hashes these normalized copies. Later report-only edits do not retroactively redefine run sources.
- `inputs/`: the seven principal native inputs and declared duration rationale.
- `native-summaries/`: run summaries, signed point-matched witnesses, temporal
  controls, and constraint/pole/lift history reductions.
- `analytic-matched-final.json`: the matched-cut strong frame-cancellation check.
- `pullback-check-final.json`, `gauge-admission.*`: independent coordinate
  pullback and unchanged-gauge admission checks.
- `active-support-admission.json`, `seed-v2-*.json`: initial lapse-only seed and
  actual reconstructed endpoint admission. The complete N12 donor range
  overlaps the support; endpoint jets/lifts are checked directly.
- `passivity-final.json`: the reproducible bitwise off/on checkpoint comparison.
- `ctest-v2.log`, `frame-regression.log`, `news-regression.log`: focused tests and
  existing extractor regressions. macOS Accelerate emitted floating-point
  warning flags despite finite, independently checked values; final scientific
  extraction uses Aurora NumPy and preserves its runtime metadata.
- `stationarity.csv`, `characteristics.csv`, accompanying provenance: pointwise
  coordinate-stationarity inventory and outgoing-speed duration estimate.
  Neither is a full evolution test.
- `pilot-diagnostics.png/.pdf`: the short N12 pilot only, through .140833 M.
  Final native plots use .718246 M and must be used for the resolution verdict.

Large accepted `.scri` records, topology epochs, initial/final checkpoints,
metric caches and full pointwise tensors remain under
`/lus/flare/projects/CompactBinaryMerger/hzhu/nonradiative-20261115`.
Collected compact/raw analysis is also at
`/Users/hz0693/research/results/nonradiative-20261115`.
Superseded local exploratory outputs are retained there in
`superseded-compact-evidence/`; they are not the final comparison.

`provenance/final-integrity-check.json` records final finite-value, source,
input and completion checks. `provenance/collector-order-check.json` verifies
that the filename-order cleanup leaves the accepted-cycle order unchanged in
all seven native archives; the N12/N16 snapshot diffs are retained.
`artifact-sha256.json` hashes this compact evidence (excluding itself).
`provenance/collected-packages.json` hashes the larger collected packages.
Scheduler totals include failed allocations; queued cancellations have no
reported execution time. Final queue snapshots are observations, not a promise
of future scheduling limits or availability.

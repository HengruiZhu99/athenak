# Full-overlap evolution policy

The hyperboloidal module now uses the validated unmasked full-overlap agreement
operator for every evolution. The no-agreement `legacy`, `masked`, and masked-band
`agreement` implementations have been removed. Ordinary AthenaK physics modules
are unaffected.

The default settings are:

```ini
<z4c_hyperboloidal>
overlap_mode = agreement_full
agreement_kappa = 1
agreement_taper_width = .015
```

The selector is optional and accepts only `agreement_full`. Explicit obsolete
selectors, `cartesian_cutoff`, `agreement_inner`, and `agreement_outer` are errors.
The inner radius comes from the atlas. Kappa and taper width must be finite and
positive. The original fixed-step admission bound remains enforced.

Cartesian and spherical rectangular overlap nodes remain evolved. The agreement
operator samples both representations, uses the existing smooth cube/radial
taper and row-mass bound, and preserves the validated donor order and optimized
reduction arithmetic. Mask-specific donor clipping, KO truncation, initial-data
subset import, remesh history routing, and AMR exclusions have been removed.

The mathematical restart identity for full overlap is unchanged. Existing
full-overlap checkpoints remain compatible when their original parameters and
mesh are retained. Checkpoints from removed policies are not migrated: explicit
old selectors are rejected, and older implicit no-agreement checkpoints fail the
restart identity check. Start from initial data to change a historical policy.

Historical comparison reports and frozen Aurora launchers remain as provenance
for their recorded binaries, not supported launch instructions for the current
executable. In particular, `hybrid_overlap_compare.py` and
`hybrid_science_segment.py` under `tst/z4c_hyperboloidal/aurora` describe retired
masked/no-agreement campaigns. Use a full-overlap input and the current binary
for new runs. The historical checkpoint diagnostic harness now accepts only
`history_output RESTART checkpoint` and evaluates the restored full-overlap plan.

## Validation, 2026-11-15

[Raw reports](evidence/full-overlap-only-20261115/) record the local MPI checks:

- All 13 CTests pass.
- Explicit and default full-overlap settings produce identical two-step N12
  payloads. The pre-cleanup binary (`9ee52c46a`) with explicit full overlap and
  the cleanup binary agree bitwise across 24,704,784 scalars. Restarting the old
  full-overlap initial checkpoint on eight ranks reproduces that same final
  payload (SHA-256 `bf13bacd165a2e77833b8e45ba70b9f2207d3919273f84fb25b343be96ee7bb1`).
- All four unsupported selectors and three obsolete mask parameters are rejected.
- Six distributed agreement-energy cases and all four native RK stages pass;
  the stage update error is exactly zero.
- AMR refinement/coarsening lifecycle tests pass on two and eight ranks. The
  shared minimum-chi/registry checks and real-state history checks pass.
- The complete scri recorder suite passes with full overlap: manufactured jets,
  topology changes, treatments B/C, empty ranks, output scheduling, invalid
  settings, enabled/disabled equality, and two-to-eight-rank restart equality.

These are CPU correctness checks on the 106-block fixture, not a claim of GPU
binary-run validation or a physical waveform extraction. The Aurora rerun uses
its own frozen source manifest and requires GPU comparisons before long evolution.
The final edit after the main checks changes comments/indentation only; a final
recorder-enabled two-step payload comparison checks the rebuilt executable too.


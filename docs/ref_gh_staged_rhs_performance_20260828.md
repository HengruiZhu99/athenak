# Ref-GH staged RHS performance qualification

## Scope and exact base

This branch starts from the accepted compact analytic radial-q checkpoint
`a09caf707f88d9fb6ca71f9abf62c9302fde3bac` on
`codex/ref-gh-analytic-device-view-performance-20260828`.  The parent worktree
was clean and local `HEAD`, its configured upstream, and the GitHub branch all
matched that commit before this separate worktree and branch were created.

Branch under development:
`codex/ref-gh-staged-rhs-performance-20260828`.

The controlling baseline is the retained ordinary-RangePolicy production path:

- 12 static plus 8 stage analytic radial-q Reals per ghosted cell;
- no generic provider/workspace/evolution/diagnostic allocation in analytic
  mode;
- `T_RefGH,RHS/T_Z4c,RHS = 10.524508196915962` from Aurora job 8789684;
- the retained primary analytic RHS kernel previously reported about 503
  spilled Reals per work item.

The rejected team-per-cell and non-inlined-team variants are not part of this
work.  Equations, q dependence, gauge driver, gamma0/gamma2, Phi ordering,
finite differences, RK4, and dissipation are frozen.

## Phase 1 local baseline

Fresh local provenance:

- source: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`;
- Kokkos: `6739bc623081648af9e752b616d9671527922cbf`, version 4.7.2;
- compiler: GCC/G++ 13.3.0;
- build: Release, Kokkos Serial, MPI/OpenMP/SYCL disabled;
- oracle executable SHA-256:
  `0915275de6cca898c51f8a163010ef819f688d607bc0d6eaa20f5231be2c9415`.

Two independent SymPy 1.14.0 regenerations took 321.99 and 323.00 seconds and
were byte-identical to one another and to the committed generated headers:

- geometry: `6b4b3976f8cfc62924aabaf6cb960cd79a09677e8a7dd59441b5d3a4f90184e7`;
- gauge: `f6ec5bb54d10c490f65f78d9d9a9e7df07c7f080eef4d6edd3fff55d45053908`;
- source: `b5dff1d44bd1e0ed53070fdd5fa4b30d0c4c176a13c2d0924dd6b0ad51012f21`;
- generator: `2c19efa4affd02c0d97ebbe26ea51a3f6bbb23467f3cb3c3e07c134320e26cc3`.

The fresh default-on source-unit executable passed without changed tolerances:

- original coefficient oracle: 216 samples, `8.88178e-15`;
- expanded coefficient oracle: 2160 samples, `1.48837e-13`;
- generated geometry: 2376 samples, `2.33147e-15`;
- moving gauge/mixed jet: 2160 samples, maximum reported motion error
  `1.24829e-14`, including finite `dtTheta`;
- compact boundary projection: 2160 samples, metric `4.56474e-14`, gauge
  `9.21858e-16`, subtracted gauge `7.11991e-16`;
- all-61 RHS: 4320 compatible and standard comparisons, `2.84217e-14`;
- generic production-cache oracle: `1.63758e-14`.

Fresh compatible and standard one-cycle runs completed for both analytic and
generic-cache backends.  Analytic allocation was exactly 1,327,104 static bytes
plus 884,736 stage bytes and zero generic bytes.  Compatible evolved rows agree
at conditioned Linf `4.32986979603811051e-15`.  Initial common-ADM shell
diagnostics retain the already documented conditioning; the two reference
Ricci columns are backend-specific and excluded from analytic/generic history
comparisons.  The all-61 source-unit oracle is the hard compatible/standard
pointwise gate.

Compact logs and histories are under
`artifacts/ref_gh_staged_rhs_performance_20260828/baseline_local/`; restart
files remain only in the temporary run directory and are not committed.

## Phase 1 Aurora status

A fresh remote source clone is staged at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_baseline_a09caf_phase1`

The first submission attempt on 2026-08-28 was rejected before a job was
created because another unrelated user job occupied the per-user queued-job
limit.  That job and its directory were not touched.  The required fresh
one/eight-tile full-output baseline, current compiler register/spill report,
and matched warmed 64-cubed benchmark therefore remain pending.

## Qualification and claim boundary

Current state:

- exact base frozen: yes;
- deterministic generation: fresh pass;
- local coefficient/geometry/mixed-gauge/boundary/all-61 gates: fresh pass;
- local analytic allocation: fresh pass;
- fresh Aurora one/eight-tile baseline: pending;
- fresh Aurora matched benchmark: pending;
- staged hot-reference or physical scratch implementation: not started;
- Ref-GH performance target: not established.

No convergence, trumpet stability, production readiness, or performance
improvement is claimed from this local checkpoint.

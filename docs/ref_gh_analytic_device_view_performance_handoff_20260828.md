# Ref-GH analytic radial-q production and performance handoff

## Final checkpoint status

Branch: `codex/ref-gh-analytic-device-view-performance-20260828`

Required starting commit:
`2d99137ca41de7df12ef1e3234f076b0ef2d8835`

This checkpoint retains the last full-output PVC-qualified compact analytic
production source from `94c66b0252510afa8f16887a1a5e18d69fcfa363`
(source-equivalent to the tested commit `77f2c8e04ff1f1b50e7a53257b707d74d283d390`).
Two later team-per-cell compiler experiments are preserved in history and
artifacts but rejected from the final source because their PVC full-output
diagnostics failed.

The generic 1171-Real reference-cache pipeline remains available only as
`generic_cache_oracle`.  The analytic production backend allocates exactly 12
static plus 8 stage Reals per ghosted cell and allocates none of
`reference_provider`, `reference_workspace`, `reference_evolution`, or
`reference_diagnostic`.

## Qualification matrix

- Analytic coefficient and generated geometry: **qualified locally**.  The
  unchanged 216-point coefficient oracle passes at `8.88178e-15`; the expanded
  radial oracle passes 2160 samples at `1.48837e-13` against `2e-13`; generated
  geometry passes 2376 samples at `2.33147e-15` against `256 epsilon`.
- Mixed-jet moving gauge path: **qualified locally**.  The 2160-sample gate
  covers `Hhat`, `dHhat`, `theta`, `dtTheta`, `L_ttr`, and `L_trr`; its largest
  reported conditioned error is `1.24829e-14` against `256 epsilon`.
- All-61 RHS equivalence: **qualified locally** for compatible and standard Phi
  ordering.  All 4320 deterministic physical states pass at
  `2.84217e-14` against `256 epsilon`, including gamma0, gamma2, the gauge
  driver, and gauge-reference subtraction.
- Analytic production backend: **integrated**.  The monolithic generated
  `ReferenceGeometry` routine and recursive spin/Riemann accessors are not used
  by CalcRHS, q measurement, physical boundaries, timestep, or ordinary
  production reference updates.
- Analytic allocation: **qualified locally**.  A 16-cubed one-block case
  reports zero generic bytes, 1,327,104 static bytes, and 884,736 stage bytes.
- Compact q estimator: **integrated**.  Closed-loop mode uses the compact
  precomputed cell/weight list and one device reduction plus one collective;
  disabled/prescribed modes do not schedule stage-by-stage measurement.
- PVC evolved cycle: **qualified for the retained production source** by job
  8789663.  Eight ranks mapped to eight distinct PVC tiles; one/eight-rank
  conditioned history error was `3.88980825583101983e-14` against `5e-12`.
- Production readiness: **not established** because the measured RHS ratio
  remains above the performance target.

## Measured performance

Aurora job 8789684 is the controlling matched warmed-up 64-cubed measurement:

- dynamic Ref-GH: `7.368637e5` active zone-cycles/s;
- static Ref-GH: `7.470968e5` active zone-cycles/s;
- Z4c: `7.230060e6` active zone-cycles/s;
- q controller: `0.628%` of complete stage, passing the 2% target;
- analytic reference update: `0.0495%`, passing the 10% target;
- dynamic/static complete-time ratio: `1.0181`, passing the 1.10 target;
- Ref-GH/Z4c complete-stage ratio: `9.1853`;
- Ref-GH/Z4c main-RHS ratio without dissipation: `10.5245`, failing the
  stretch objective of at most 2.

The compact physical boundary is `7.95%` of the dynamic stage and is 79.22
times faster than the previous generic projected boundary.  No later
performance result supersedes job 8789684.

## Rejected compiler experiments

The first team-per-cell source candidate at `441a6c05` passed all local oracles
and evolved comparisons.  Job 8789799 failed because Kokkos AUTO selected a
work-group above PVC's 512-work-item limit.  The explicit 16-lane correction at
`17b52c0f` allowed one- and eight-rank cycles to complete in job 8789862, but
native GH/reduction/curl histories were nonfinite from the initial row.  The
team kernel still spilled about 1150 Reals.

The follow-up `9867f1d6` retained the accepted joint-CSE matrix algebra, moved
the large contractions behind non-inlined device call boundaries, and stored
only two rank-2 contractions in per-team scratch.  It passed deterministic
regeneration, every local oracle, and a local evolved analytic/generic
comparison.  Job 8789983 nevertheless reproduced the nonfinite native
diagnostics, and the team kernel spilled about 1257--1263 Reals.  Both physical
field outputs remained finite and agreed at printed precision, but that does
not satisfy the full-output gate.  No benchmark followed either failure.

The production source was therefore restored exactly to the qualified compact
boundary implementation.  The failed candidates remain reviewable in Git
history; they are not active production dispatch.

## Evidence and remote artifacts

Compact local evidence is under
`artifacts/ref_gh_analytic_radial_q_20260828/`.  Controlling files include:

- `local_checkpoint.txt`;
- `compact_boundary_pvc_performance_20260828.txt`;
- `compact_boundary_performance_8789684_analysis.json`;
- `team_per_cell_pvc_8789799_failure.txt`;
- `team_per_cell_pvc_8789862_failure.txt`;
- `noninline_source_local_checkpoint.txt`;
- `noninline_source_pvc_8789983_failure.txt`.

Large restart files were not committed.  They remain at the remote run paths
recorded in each failure report, with SHA-256 manifests sufficient to identify
them.  In particular, job 8789983 data are under:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_noninline_source_20260828_9867f1d6_gate1/runs/analytic_q_pvc_8789983.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

## Claim boundary and next review

This checkpoint qualifies the compact coefficient/geometry algebra, the mixed
gauge jet, all-61 RHS equivalence, analytic allocation, retained compact
production dispatch, and its bounded eight-tile PVC cycle.  It does **not**
qualify the rejected team-per-cell implementations, the `<=2x` RHS objective,
overall production readiness, convergence, trumpet formation, or long-time
stability.

A follow-up performance/code review should begin from the retained source and
the compiler/profiling evidence, not by re-enabling either rejected team
candidate.  The main open issue is to reduce the compact primary RHS cost
without expanding persistent storage, materializing spin/Riemann caches,
reconstructing physical geometry in multiple kernels, or weakening the
full-output PVC gate.

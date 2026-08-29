# Phase 2 matched-medium attempt: Aurora job 8790836

This is a failed and preserved production attempt, not long-time evidence.

## Configuration

- Source commit: `4d1ede077011eeb4d25c0cb983f5d56cd381b7ff`
- Frozen production `src/`: byte-identical to `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`
- Executable SHA-256: `b247762c83e5bba2b8a5331f9ce372d17462eb6f7ef59c63b4ddef39306a05e0`
- Aurora queue/project: `debug-scaling` / `CompactBinaryMerger`
- Grid: uniform 96^3 over `[-2M,2M]^3`, 216 16^3 MeshBlocks,
  `h=M/24`, four ghost cells
- Decomposition: 216 MPI ranks on 18 nodes, one MeshBlock and one distinct PVC
  tile per rank
- Physics/numerics: exact physical stationary trumpet, analytic reference
  `q=1`, q controller off, compatible Phi ordering, FD4, RK4, CFL 0.05,
  gamma0=gamma2=1, dissipation 0.02, exact projected trumpet boundaries
- Requested endpoint: `t=100M`

## Result

The trajectory failed closed after cycle 330 at `t=1.123484M` with
`ref_gh reached an invalid effective timestep`.  PBS exit status was 143.  The
latest accepted history row is `t=1.000922M`; the only restart is the initial
`t=0` checkpoint, so this trajectory cannot be continued.

The growth is visible well before the fatal guard.  Puncture-stencil-excluded
GH RMS grew from `5.91e-14` at t=0 to `3.60e-4` at t=1.0009, while the relative
metric condition diagnostic grew only to `1.0000487` at the last accepted
history.  The source-frame diagnostic grew from `7.46e-11` to `4.87e3` and
the Q/Delta diagnostic maxima grew to 126/78.4.  Thus this was not a scheduler
timeout or an output-only failure.

The offline binary64 health summary covers only the initial snapshot because
the run never reached the 2M output cadence.  Its `all_pass=true` must not be
misread as a passed trajectory.

## Interpretation and bounded discriminator

This attempt used a new one-MeshBlock-per-rank decomposition, whereas the
previous stationary evidence used many MeshBlocks per rank.  The failure is
therefore not yet attributable uniquely to the mathematical formulation.  A
single bounded rerun will retain the identical physical grid and all equation/
time-integration parameters while using the established 12-tile pattern (18
MeshBlocks per rank).  If the same growth remains, Phase 2 fails as a physical
trajectory.  If it disappears, job 8790836 remains a genuine decomposition/
communication portability failure and cannot be used for production scaling.

No profiler was run after this invalid numerical trajectory.  The 216-rank
attempt advanced only about 330 cycles in 30.6 seconds, approximately
`9.7e6` active zone-cycles/s in aggregate; this already shows that using 216
tiles did not provide useful strong scaling relative to prior compact runs.

## Large scratch evidence

Uncommitted restart and binary field payloads remain under:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_single_puncture_robustness_20260829_253c78e2/runs/phase2_matched_q100_h24_8790836.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

Compact histories, mappings, logs, hashes, health JSON, provenance, and the
final PBS record are preserved alongside this report.

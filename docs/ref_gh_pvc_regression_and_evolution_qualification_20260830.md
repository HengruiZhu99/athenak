# Ref-GH PVC regression and evolution qualification

Status: `PVC PORTABILITY REGRESSION UNRESOLVED`

This report tracks the source-regression and GPU-lifecycle audit required
before the fully-subtracted stationary-trumpet evolution campaign can resume.
It is deliberately claim-limited: local CPU success and cycle-zero PVC output
do not qualify positive-time evolution.

## Frozen source checkpoints

| Label | Commit | Tree | Role |
| --- | --- | --- | --- |
| W | `68bc1ee30c0ac64c3afe8d3961bead65efce205a` | `00d370276ed4b1377b2b9e46456f838e8e4cbbb0` | Last known Aurora-positive Ref-GH source |
| R | `3c9a34c8c3123c2570eb33e8ec77368feb1f1c61` | `677936216ba8f81d885563bcde41b6596f31cc08` | First repaired source associated with the new PVC failure |
| C | `5e36b641b1ff04272b59e9a66f35db4d1a4ac363` | `d8f0428f7d7196b34766b27435c76613f12603a6` | Starting repair tip with later portability discriminators |

All three source checkpoints pin Kokkos commit
`6739bc623081648af9e752b616d9671527922cbf`.

## Phase 1: identical-build W versus R

The controlling job is
`scripts/ref_gh/aurora_pvc_regression_wr_control.pbs`. It configures and builds
W and R sequentially on one Aurora node, with one module environment and
Kokkos bounds checking disabled, then runs the frozen 96^3, 216-MeshBlock,
12-rank STANDARD Case-D workload for exactly one RK cycle on that same node.

Result: pending.

## Remaining required evidence

The first-bad source bisect, hybrid matrix, one-rank comparison, explicit MPI
completion discriminator, rank-packed message-size audit, optional device-code
split discriminator, minimal permanent correction, repeated one-cycle PVC
qualification, and all positive-time 3M/5M/resolution/20M/100M results remain
unexecuted. No stationary-trumpet robustness or convergence claim is made.

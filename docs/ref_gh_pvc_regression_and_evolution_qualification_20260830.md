# Ref-GH PVC regression and evolution qualification

Status: source regression established; first-bad commit bisect in progress.

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

Aurora job `8791722` ran on node `x4216c4s7b0n0` with twelve distinct PVC
tile mappings. Both builds used Intel oneAPI DPC++ 2025.3.2, MPICH
5.0.0.aurora_test.3c70a61, Kokkos SYCL/PVC, GPU-aware MPI, Release mode, and
Kokkos bounds checking off. The controlling input SHA-256 was identical:
`6d483ded11b70d640f4a166fd21757f956802622d4e3994ceda71ae8649235eb`.

| Source | Executable SHA-256 | Result | Latest history time |
| --- | --- | --- | --- |
| W | `e0f2b1e1380b3bc4c3d296ef77e8a5af39bc426c392c8d625de5342b245f585a` | `PASS_ONE_CYCLE_PVC` | `0.003404497M` |
| R | `ae105aa20563ab3d7f9020bc4966cfec244ffcbc667d7caa064ecf42165c5132` | `FAIL_LEVEL_ZERO` | `0M` |

W completed the positive-time cycle with finite diagnostic output. R passed
through its first physical metric and gauge boundary fences, then multiple
tiles reported Level Zero PDE-level `NotPresent` writes and the MPI job ended
with status 143. Since W passed and R failed on the same node and contemporary
build environment, a source regression is established. The last reported
kernel remains only a localization boundary, not proof of the corrupting
operation.

Compact evidence is under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase1_aurora_8791722_wr_control`.
The full Aurora campaign remains at
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_regression_20260830_wr_v2`.

## Remaining required evidence

The first-bad source bisect, hybrid matrix, one-rank comparison, explicit MPI
completion discriminator, rank-packed message-size audit, optional device-code
split discriminator, minimal permanent correction, repeated one-cycle PVC
qualification, and all positive-time 3M/5M/resolution/20M/100M results remain
unexecuted. No stationary-trumpet robustness or convergence claim is made.

# Evidence disposition manifest

Every file is hashed by `SHA256SUMS`.  This document states which artifacts are
authoritative results and which are preserved failed attempts.

## Phase 0 — base authority

| artifact | disposition |
|---|---|
| `evidence/phase0/dynamic-wave-2d-n16-n32.log` | authoritative base 2D negative-order failure |
| `evidence/phase0/dynamic-wave-3d-n16-n32.log` | authoritative base 3D negative-order failure |
| `evidence/phase0/dynamic-wave-*-n64-amr.log` | base N64 run logs; exact offline comparison values are summarized in reports |
| `evidence/phase0/dynamic-wave-3d-n64-uniform-corrected.log` | valid uniform N64 3D control |
| `evidence/phase0/cc-exact-regression.log` | authoritative CC exact-selection/fingerprint pass |

The initially mislabeled 3D N64 “uniform” run actually enabled AMR and was
removed; it is not part of the manifest.

## Phases 1–3 — localization and repair development

| artifact group | disposition |
|---|---|
| `evidence/phase1/*first_stage_v2*` | authoritative T-stage contributor localization before floor fix |
| `evidence/phase1/*post_floor_fix*` | floor-index ablation; shows remaining order limitation |
| `evidence/phase1/*.csv.gz` | compressed raw default-off state/RHS contributor records |
| `evidence/phase2/q8-axis-pre-rhs-state.csv.gz` | raw q8/axis state diagnostic |
| `evidence/phase2/q8-buffer-cardinality-fixed.log` | intermediate cardinality repair evidence |
| `evidence/phase3/*` | coarse-halo/restart localization and repaired restart evidence |

Intermediate failures are retained as causal evidence, not counted as final
passes.

## Phase 4 — host qualification

Authoritative results:

- `dynamic-wave-o2-rk4-{2d,3d}-n16-n32-n64.log`;
- `dynamic-wave-{2d,3d}-n16-n32-n64.log` (O4);
- `dynamic-wave-o6-rk4-{2d,3d}-n24-n36-n48.log`;
- `release-host-final-post-mpi-range-fix-ctest.log`;
- `post-mpi-range-fix-focused-ctest.log`;
- `debug-post-mpi-range-fix-focused-ctest.log`;
- `debug-post-ownership-restart-teardown.log`.

Qualified limitation:

- `dynamic-wave-o6-rk4-2d-n24-n48-n96.log` is a roundoff-saturation control,
  not a valid N48/N96 convergence claim.

Preserved non-source failures:

- `debug-asan-ubsan-focused-ctest.log` records a 45-second subprocess timeout;
  `debug-asan-ubsan-restart-3d-manual.log` shows the unchanged case passing with
  a 300-second timeout;
- `debug-post-pmrc-restart-3d-manual-v2.log` exposes the subsequent 8-byte
  `ProblemGenerator` restart-buffer leak after the AMR-owner leak was fixed;
- `debug-post-pgen-restart-3d-manual.log` is empty because the full wrapper was
  terminated by its outer six-minute timeout; it is orchestration evidence,
  not a numerical failure or pass;
- `release-mpi-focused-ctest.log` records a workstation MPI/DRM environment
  hang before Athena startup.

The Kerr logs record an expected fail-closed rejection at a true vertex
puncture, not a successful black-hole evolution.

## Phase 5 — Aurora/PVC

| bundle | disposition |
|---|---|
| `aurora-build/` | authoritative source/build/device provenance for `21b91213` |
| `aurora-qualification-v1/` | numerical dynamic/restart/output passes plus nine `h5py` harness failures; wrapper status 8 is not a numerical verdict |
| `aurora-qualification-v2-remainder/` | authoritative `h5py`-enabled static/gauge passes and 2/4-rank Cartoon passes; wrapper status 1 is caused by an invalid one-root-block/four-rank 3D fixture |
| `aurora-qualification-v3-mpi-completion/` | authoritative 2-rank 3D pass and four-rank 3D PVC page fault that exposed the remote halo OOB |
| repaired job `8775888` | submitted with executable hash recorded in `BACKEND_MATRIX.md`; no result collected after authentication failure, therefore not evidence of a pass |

Large restart payloads are stored as lossless `.rst.gz` archives.  The captured
Aurora bundle manifests preserve hashes of the original uncompressed files;
the top-level `SHA256SUMS` validates the committed compressed representation.

## Open evidence

- current-source Perlmutter CUDA build/runtime/restart/memory check;
- current repaired-source fixed-grid and common-tree Brill tests;
- regular true-vertex black-hole evolution;
- performance profiles and matched CC/VC throughput.

No file from an older Aurora or Perlmutter campaign is silently promoted to
current-source evidence.

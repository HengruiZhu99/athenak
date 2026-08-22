# Native vertex-centered vacuum Z4c hardening report

Date: 2026-08-22

## Verdict

`VC_Z4C_NOT_QUALIFIED`

The original Aurora/SYCL AMR page fault is repaired and the supported native
vertex-centered infrastructure now has broad 2D/3D host and PVC lifecycle
coverage. Qualification nevertheless stops fail-closed because:

1. a nonconstant dynamic-AMR wave mismatch grows with N in both 2D and 3D;
2. exact-final SYCL does not reproduce the 3D restart payload bit-for-bit;
3. a current-source CUDA runtime matrix is unavailable.

These are separate from the fixed host-pointer capture. No interpolation,
admissibility, floor, gauge, KO, CFL, or formulation threshold was weakened.

## Root cause and repair

The pre-fix `MeshBoundaryValuesVC::ProlongateVC` device lambda read
`recvbuf[n].iprol[0]`, where `recvbuf` is host-side boundary bookkeeping. The
PVC kernel therefore dereferenced a host pointer at the first coarse/fine
boundary prolongation after AMR. Commit `aa668dd0` copies those ranges to the
device-resident `prolongation_bounds` DualView during buffer initialization and
captures only its device view. Aurora job `8775368` passes the complete A0--A19
and A150--A159 canary sequences repeatedly. See `ROOT_CAUSE.md`.

## Infrastructure qualification obtained

- deterministic 2D Cartesian, 2D axis-touching Cartoon, and 3D Cartesian
  O2/O4/O6 refine/derefine lifecycles pass on host and exact-final PVC;
- collapsed directions retain one point and zero ghost-width contribution;
- 3D face/edge/corner and octant paths pass;
- static interfaces, axis parity/regularity, shared-node synchronization,
  history quadrature, restart/replay, and output contracts pass on host;
- analytic gauge waves converge at expected O2/O4/O6 rates in both CC and VC,
  2D and 3D;
- the complete Release enabled suite passes 121/121, with only two
  backend-required CUDA tests disabled;
- the historical cell-centered payload remains exact.

## Remaining numerical failure

The nonconstant wave discriminator applies a deterministic physical-time
refine/derefine schedule and compares N16 and N32 accepted fields. Its errors
increase rather than decrease:

| geometry | N16 RMS | N32 RMS | order |
|---|---:|---:|---:|
| 2D Cartesian | `1.185856836e-6` | `2.289381583e-6` | `-0.9490` |
| 3D Cartesian | `1.676664824e-7` | `2.283913019e-7` | `-0.4459` |

The refinement transaction itself is geometrically correct. The first large
discrepancy appears after the first evolved fine stage at an interface. A
coarsest-level synchronization experiment improved N16-to-N32 but diverged at
N64 and at amplitude `1e-6`; it was fully reverted. The evidence does not
justify a further production change. It localizes the remaining problem to
nonconstant persistent/dynamic VC AMR behavior, not to the original device
metadata lifetime.

## Cell-centered preservation

The implicit default and explicit `grid_centering=cell` produce exact history,
timestep, binary arrays, and restart payloads. A separately built candidate is
also exact against historical authority `6daa774d...`. CC lifecycle companions
pass in all three supported geometries. See `CC_REGRESSION.md`.

## Restart/output/backend limitation

Host 2D/3D restart, output, history, and replay gates pass. Exact-final PVC
passes all nine lifecycle tests, both 2D restart tests, and all three output
tests. Its 3D post-refinement restart continuation has the same times,
hierarchy, and timestep but a non-identical final binary payload. This blocks
the exact restart gate. Current-source CUDA runtime evidence is absent because
Perlmutter authentication was unavailable. See `BACKEND_STATUS.md`.

## Prior bounded Brill evidence

The pre-existing bounded fixed-grid O4 VC Brill matrix remains valid: N128,
N256, and N512 reach `t=0.5 M`, with nontrivial evolved-field orders
3.314--4.081. It is not rerun because the new nonconstant infrastructure gate
fails before Phase 10.

The authenticated common-tree Brill discriminator remains negative. Exact
zero-ULP hierarchy replay does not produce convergent evolution; higher
resolution loses metric SPD earlier and constraint convergence is negative in
the common admissible interval. The controlling summary is
`docs/investigations/brill_o4_dchi001_replay_convergence_20260821/comparison_summary.json`
with `bulk_convergence=DIVERGES_WITH_RESOLUTION` and
`overall=O4_NONCONVERGENT`. No Figure-3, critical-collapse, horizon, long-time,
or production-readiness claim is made.

## Acceptance table

| gate | disposition | evidence boundary |
|---|---|---|
| `vc_amr_root_cause` | PASS | host metadata capture proven; device DualView repair; Aurora A0--A19/A150--A159 pass |
| `vc_2d_cartesian` | FAIL | O2/O4/O6 lifecycle passes, but bounded nonconstant convergence is negative |
| `vc_2d_cartoon` | PASS (infrastructure) | O2/O4/O6 lifecycle, axis parity/regularity, quadrature pass |
| `vc_3d_cartesian` | PASS (lifecycle) | O2/O4/O6 face/edge/corner lifecycle passes; nonconstant discriminator remains negative |
| `vc_same_rank_exchange` | PASS | exact cursor/count and shared-node tests pass |
| `vc_mpi_exchange` | PASS selected | exact-final two-rank PVC 2D Cartoon O4 and 3D Cartesian O4 lifecycles pass in job `8775545` |
| `vc_static_amr` | PASS | 2D Cartesian, axis-touching Cartoon, and 3D static interfaces pass |
| `vc_restart` | FAIL backend-wide | host 2D/3D pass; SYCL 3D exact payload mismatch |
| `vc_output_history` | PASS on host/SYCL selected | 2D/3D nodal output and ring history contracts pass |
| `vc_amr_replay` | PASS on host | 2D/3D event/tree replay and restart continuation pass |
| `vc_gauge_wave` | PASS on host | real 2D/3D CC/VC O2/O4/O6 evolution and convergence |
| `cc_selector_equivalence` | PASS | implicit/explicit selector exact payloads |
| `cc_full_regression` | PASS | candidate exact against historical CC authority |
| `whole_code_regression` | PASS on primary host | 121/121 enabled; 2 CUDA-required disabled |
| `cuda_exact_final` | PENDING | current Perlmutter runtime unavailable; no inferred pass |
| `sycl_exact_final` | FAIL | lifecycle/output largely pass; 3D restart exactness fails |
| `brill_fixed_grid` | PASS, bounded prior evidence | O4 through `t=0.5 M` only |
| `brill_common_tree` | FAIL | exact replay but resolution-worsening SPD/constraint behavior |
| `overall` | `VC_Z4C_NOT_QUALIFIED` | required numerical, restart, and backend gates remain open |

## Evidence locations

- compact retained evidence: this directory and `evidence/local/`
- repaired Aurora canary:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-hardening-829be2f6-v1-20260822/campaign-device-metadata-repair-v2`
- exact-final Aurora matrix:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-hardening-99a4eb5b-v2-20260822/campaign-exact-final-sycl-v1`
- exact-final Aurora harness retry:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-hardening-99a4eb5b-v2-20260822/campaign-exact-final-sycl-v2`
- exact-final Aurora two-rank matrix:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-hardening-99a4eb5b-v2-20260822/campaign-exact-final-sycl-v3-mpi`
- local exact Aurora canary copy:
  `/home/hzhu/Desktop/research/gr/collapse/artifacts/z4c_vc_hardening_20260822/aurora/campaign-device-metadata-repair-v2`

Large build trees, executables, and raw restarts are intentionally not
committed. Their exact hashes and external paths are recorded in the manifest.

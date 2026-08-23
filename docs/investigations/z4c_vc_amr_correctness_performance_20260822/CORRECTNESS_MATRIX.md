# Correctness matrix

Status is for the numerical implementation through
`480de5f7bd7d510d9c5984a5e5b7dcbf60d2b3a2`, with ownership cleanup at
`144aaded1c56f4443d1e4d31f45c0168ab849f06`.

## Host Release/OpenMP

| gate | 2D Cartesian | Cartoon axis | 3D Cartesian | evidence/status |
|---|---|---|---|---|
| layout, coordinates, collapsed dimensions | pass | pass | pass | final CTest |
| FD kernels O2/O4/O6 | pass | pass | pass | final CTest |
| transfer weights/injection/tensors/bounds | pass | pass | pass | `athena.z4c_vc_amr_transfer` |
| dynamic refine/evolve/derefine O2/O4/O6 | pass | pass | pass | final CTest |
| static multilevel O4 | pass | pass | pass | final CTest |
| RK1/RK2/RK3/RK4 lifecycle | pass (O4) | RK4 pass | RK4 pass | only 2D Cartesian spans RK1-RK4 |
| fixed-grid gauge wave O2/O4/O6 | pass | not an analytic fixture | pass | final CTest |
| output/history/replay | pass | pass | pass | final CTest |
| exact restart continuation | pass | pass | pass | final CTest |
| CC exact regression | pass | pass where covered | pass where covered | selector and production fingerprints exact |
| ASan/UBSan/Kokkos debug | pass | pass | bounded pass | dynamic 3D and direct restart teardown clean; full post-cleanup restart wrapper timed out externally |

Final host result: 126/126 enabled tests passed.  Two CUDA-required tests were
disabled by configuration and are not counted as passes.

## Dynamic nonconstant AMR convergence

RMS and Linf values are `(RMS, Linf)` differences between matched uniform and
refine/evolve/derefine runs.

### O2/RK4

| geometry | N16 | N32 | N64 | RMS orders |
|---|---:|---:|---:|---:|
| 2D | `(5.69273e-8, 4.13199e-7)` | `(1.33301e-8, 9.86570e-8)` | `(2.57330e-9, 1.92427e-8)` | `2.094, 2.373` |
| 3D | `(7.78996e-8, 3.99565e-7)` | `(1.74606e-8, 9.57200e-8)` | `(3.35449e-9, 1.92844e-8)` | `2.158, 2.380` |

### O4/RK4

| geometry | N16 | N32 | N64 | RMS orders |
|---|---:|---:|---:|---:|
| 2D | `(2.14224e-9, 1.55803e-8)` | `(1.23446e-10, 9.09652e-10)` | `(5.23471e-12, 3.87572e-11)` | `4.117, 4.560` |
| 3D | `(2.92597e-9, 1.52382e-8)` | `(1.62958e-10, 9.01658e-10)` | `(6.89207e-12, 4.15352e-11)` | `4.166, 4.563` |

### O6/RK4

The q8 single-hop halo contract makes N16 with two root MeshBlocks per active
direction invalid.  N24/N36/N48 use valid MeshBlock geometry.

| geometry | N24 | N36 | N48 | RMS orders |
|---|---:|---:|---:|---:|
| 2D | `(4.54184e-12, 3.34151e-11)` | `(5.51294e-13, 4.07964e-12)` | `(8.57347e-14, 6.53317e-13)` | `5.201, 6.469` |
| 3D | `(6.04922e-12, 3.40285e-11)` | `(7.46038e-13, 4.15686e-12)` | `(1.12792e-13, 6.41171e-13)` | `5.162, 6.567` |

The N48/N96 O6 control reaches floating-point saturation and is not used as a
convergence interval.

## Baseline-to-repair discriminator

| geometry | baseline N16 RMS | baseline N32 RMS | baseline order | repaired O4 N16->N32 order |
|---|---:|---:|---:|---:|
| 2D | `1.185856836e-6` | `2.289381583e-6` | `-0.9490` | `4.1172` |
| 3D | `1.676664824e-7` | `2.283913019e-7` | `-0.4459` | `4.1663` |

The recorded baseline N64 RMS values were approximately `2.906e-6` (2D) and
`2.862e-7` (3D), continuing the worsening trend.  These approximate N64 values
are included for localization context; the exact N16/N32 failure tuples above
are the preserved authoritative baseline log.

## MPI and ordering

Host MPI execution could not be qualified on this workstation: even
`mpiexec -n 2 /bin/hostname` hung in the local DRM scheduler before Athena
started.  This is recorded as an environment failure, not a numerical failure.

Aurora/PVC current-source evidence covers 2-rank and 4-rank Cartoon O4
refine/derefine.  A two-rank 3D O6 pre-repair run passed; the pre-repair
four-rank run exposed the migration-range defect.  The repaired-source job and
3D rank-change restart result could not be retrieved and remain open in
`BACKEND_MATRIX.md`.

## Physical gates

| test | status | reason |
|---|---|---|
| fixed-grid Brill, current repaired source | open | not rerun on current source |
| common-tree Brill convergence | open | prior exact replay diverged with resolution; repair not yet replayed |
| early constraint/SPD diagnostics | open | require the current physical replay |
| vertex-centered Kerr puncture at `r=0` | expected rejection | analytic ADM carrier diverges at the puncture; no clipping allowed |
| regular short VC black-hole fixture | open | no current pgen supplies regular true-vertex data for this gate |

Therefore the synthetic AMR correction is numerically supported, but the full
correctness qualification requested by the goal remains open.

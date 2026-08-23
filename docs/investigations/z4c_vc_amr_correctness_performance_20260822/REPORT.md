# Native vertex-centered Z4c AMR correctness handoff

Date: 2026-08-22

Repository: `https://github.com/HengruiZhu99/athenak`

Branch: `codex/z4c-vc-amr-correctness-performance-20260822`

Exact base: `34cd6db227e81aed91064f3684733e720c9fac7e`

Final source commit: `144aaded1c56f4443d1e4d31f45c0168ab849f06`

Evidence archive commit: `ac996e618ffc084fdb6758d7df0e4009d92a12e0`

Implementation commits:

- `21b9121339185ba2629ead53b6993596bbc64b62` — native-VC interpolation,
  ownership, halo, boundary, and restart repair;
- `cb4f173b5326302431c1d18e4edc113750471cef` — broadened durable test matrix
  and valid multirank fixture geometry;
- `480de5f7bd7d510d9c5984a5e5b7dcbf60d2b3a2` — bounded remote
  child-migration refinement halo;
- `144aaded1c56f4443d1e4d31f45c0168ab849f06` — release of the AMR
  criterion owner and temporary restart buffers found by LeakSanitizer.

## Final verdicts

```text
VC_Z4C_CORRECTNESS_QUALIFIED = NO
VC_Z4C_BACKEND_QUALIFIED     = NO
VC_Z4C_PERFORMANCE_QUALIFIED = NO
VC_Z4C_PRODUCTION_QUALIFIED  = NO
```

The narrow synthetic native-VC AMR defect is repaired and demonstrates the
expected O2/O4/O6 behavior on host.  Full qualification fails closed because
current-source CUDA, repaired-source physical Brill/common-tree evidence, a
regular true-vertex black-hole fixture, and performance evidence are absent.
Any unresolved Aurora completion row is also listed explicitly below.

## Root cause and first failing checkpoint

The first bad operation before repair was T5 lower-side coarse-to-fine ghost
prolongation.  C++ signed division mapped negative odd fine offsets to the wrong
coarse interval; T7 then produced an approximately `O(1/h)` `Atilde_zz` RHS
mismatch.  Once floor indexing was corrected, nominal-order interpolation still
gave only about second-order interface behavior because Z4c second derivatives
lower an `O(h^q)` ghost interpolation error to `O(h^(q-2))`.  Transfer orders
are therefore q4/q6/q8 for O2/O4/O6 bulk stencils.

Four-rank 3D/PVC testing subsequently exposed an independent remote-migration
out-of-bounds range: q8 child packing used coarse-cache width 5 in the fine
source array, while only refinement halo 3 is valid.  The upper-child range was
`7..25` for fine storage `0..24`.  The corrected remote path now uses the same
`q/2-1` halo as same-rank refinement and validates source/target storage bounds
on host before launching device kernels.

Full derivations and stage evidence are in [ROOT_CAUSE.md](ROOT_CAUSE.md),
[TRANSFER_OPERATOR_DERIVATION.md](TRANSFER_OPERATOR_DERIVATION.md), and
[AMR_STAGE_ORDER.md](AMR_STAGE_ORDER.md).

## Key numerical evidence

The original O4 dynamic-AMR discriminator worsened from N16 to N32:

| geometry | N16 RMS | N32 RMS | order |
|---|---:|---:|---:|
| 2D | `1.185856836e-6` | `2.289381583e-6` | `-0.9490` |
| 3D | `1.676664824e-7` | `2.283913019e-7` | `-0.4459` |

After repair:

| order/geometry | coarse, medium, fine RMS errors | observed orders |
|---|---|---|
| O2 2D N16/32/64 | `5.693e-8, 1.333e-8, 2.573e-9` | `2.094, 2.373` |
| O2 3D N16/32/64 | `7.790e-8, 1.746e-8, 3.354e-9` | `2.158, 2.380` |
| O4 2D N16/32/64 | `2.142e-9, 1.234e-10, 5.235e-12` | `4.117, 4.560` |
| O4 3D N16/32/64 | `2.926e-9, 1.630e-10, 6.892e-12` | `4.166, 4.563` |
| O6 2D N24/36/48 | `4.542e-12, 5.513e-13, 8.573e-14` | `5.201, 6.469` |
| O6 3D N24/36/48 | `6.049e-12, 7.460e-13, 1.128e-13` | `5.162, 6.567` |

The O6 N48/N96 control is roundoff-saturated and is not used for a convergence
claim.  Exact tuples including Linf values are in [CORRECTNESS_MATRIX.md](CORRECTNESS_MATRIX.md).

## Host qualification

- Release/OpenMP: the complete post-distributed-fix matrix passed all 126/126
  enabled tests; the two disabled cases require CUDA.
- The distributed-halo repair also passed focused 3D O6 AMR, exact 3D restart,
  transfer, CC selector, and CC production-fingerprint tests.
- ASan/UBSan/Kokkos-debug dynamic 3D AMR passed.  LeakSanitizer then exposed a
  240-byte `MeshRefinement::pmrc` ownership leak and an 8-byte restart-buffer
  leak.  Both were repaired; a direct current-source restart/teardown check is
  clean.  The full expensive debug restart wrapper later reached its outer
  six-minute orchestration timeout and is not promoted as a completed matrix.
- Implicit and explicit CC selection are exact; O2/O4/O6 production state,
  diagnostic, history, and waveform fingerprints remain exact.

## SYCL/PVC qualification

Current-source single-rank O2/O4/O6 dynamic AMR in 2D Cartesian, Cartoon, and
3D passes on an Intel Data Center GPU Max 1550 through Level Zero.  Static
multilevel and gauge-wave tests also pass after loading the missing `h5py`
module.  Exact 2D/Cartoon/3D restart and output tests pass.

Two-rank and four-rank Cartoon O4 pass.  A valid two-rank 3D O6 case passes.  A
valid four-rank 3D O6 case exposed the remote refinement-halo page fault
described above.  A repaired-source job (`8775888`) was submitted with exact
executable hash
`80b59c0cbe6cd04f283366493e6052a1ba819260dfef345096c79e8a79a29bf1`,
but its terminal status and output could not be retrieved after Aurora began
rejecting this session's login certificate.  Therefore repaired four-rank 3D
and rank-change restart remain open rather than being inferred from submission.

The first Aurora wrapper's status 8 was an analysis dependency error
(`ModuleNotFoundError: h5py`), not a numerical failure; the rerun with
`py-h5py/3.14.0` passed all nine affected tests.

See [BACKEND_MATRIX.md](BACKEND_MATRIX.md) and
[RESTART_ANALYSIS.md](RESTART_ANALYSIS.md).

## CUDA and physical evidence

Perlmutter login is blocked in this session by SSH public-key/certificate
rejection before allocation.  Aurora access was likewise unavailable at final
collection time.  No current-source CUDA build or runtime was performed, and
no CUDA result is inferred from host or SYCL.

No current repaired-source fixed-grid/common-tree Brill run was executed.  The
prior common-tree O4 campaign achieved exact hierarchy replay but diverged with
resolution; it is historical motivation, not qualification evidence for this
branch.  Moving those artifacts from Aurora to Perlmutter would aid staging but
would not replace a current-source rerun.

The existing puncture pgen correctly rejects a true vertex grid containing
`r=0`, where its analytic ADM carrier diverges.  No epsilon clipping or floor
was introduced.  A regular true-vertex black-hole fixture remains open.

## Performance

Performance work was not authorized by the gate sequence because CUDA and
physical correctness remain open.  No CC/VC ratio, profile, or optimization
claim is made.  See [PERFORMANCE.md](PERFORMANCE.md).

## Evidence map and limitations

- `evidence/phase0/`: exact base failures and CC authority;
- `evidence/phase1/`: T0-T9/state/RHS localization and floor-index ablation;
- `evidence/phase2/`: q-order/halo development evidence;
- `evidence/phase3/`: restart/coarse-halo repair evidence;
- `evidence/phase4/`: host convergence, sanitizers, restarts, and full CTest;
- `evidence/phase5/`: Aurora build/runtime/MPI evidence;
- `SHA256SUMS`: strict local artifact manifest.

This handoff does not claim Brill convergence, Figure 3 reproduction, critical
collapse, horizon formation, CUDA support, VC performance parity, or production
readiness.

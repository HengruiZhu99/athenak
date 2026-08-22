# Test matrix

Production source authority for compiled results:
`99a4eb5ba7713f7de73239cf75a27c1fb9ac6cbb`, tree
`e8c1083cc9ea67aa4a3a2c3adbffb9c31fe32c83`, Kokkos
`6739bc623081648af9e752b616d9671527922cbf`.

## Host production-path matrix

| Geometry | Order | Centering | Ranks/backend | Dynamic lifecycle | Static/interface | Gauge wave |
|---|---:|---|---|---|---|---|
| 2D Cartesian | 2 | VC | 1/OpenMP | pass | derivative pass | order 1.9646 |
| 2D Cartesian | 4 | VC | 1/OpenMP | pass | interface RMS `2.2902e-5`; interior `2.2715e-8` | order 3.9771 |
| 2D Cartesian | 6 | VC | 1/OpenMP | pass | derivative pass | order 5.9659 |
| 2D Cartoon, axis touching | 2 | VC | 1/OpenMP | pass | parity/axis pass | not applicable |
| 2D Cartoon, axis touching | 4 | VC | 1/OpenMP | pass | interface RMS `9.2133e-14`; axis `4.8016e-30` | not applicable |
| 2D Cartoon, axis touching | 6 | VC | 1/OpenMP | pass | parity/axis pass | not applicable |
| 3D Cartesian | 2 | VC | 1/OpenMP | pass | face/edge/corner lifecycle pass | order 1.9646 |
| 3D Cartesian | 4 | VC | 1/OpenMP | pass | static multilevel pass | order 3.9771 |
| 3D Cartesian | 6 | VC | 1/OpenMP | pass | face/edge/corner lifecycle pass | order 5.9659 |

The corresponding 2D/3D cell-centered gauge-wave orders are 1.9646,
3.9771, and 5.9659. Three CC lifecycle companions also pass.

Exact-final Aurora/PVC two-rank O4 lifecycle cases pass for axis-touching 2D
Cartoon and 3D Cartesian in job `8775545`. This closes the selected MPI
exchange gate; it does not alter the separate restart or numerical verdicts.

## Full and instrumented builds

| Build | Result | Exact evidence |
|---|---|---|
| Release OpenMP+Serial | 121/121 enabled pass; 2 CUDA-required disabled | `evidence/local/release-ctest.log` |
| Release MPI+OpenMP | configure and full executable build pass | executable `584b3e3b...`; runtime delegated to Aurora |
| Debug OpenMP, Kokkos bounds + DualView checks | 120 enabled tests pass in the concurrent run; isolated 3D VC history/replay passes in 313.49 s; generic AMR-history and full convergence are instrumentation-time limitations, not assertions | cache `c321803e...` |
| ASan+UBSan OpenMP | build pass; selected lifecycle/restart/output matrix passes after allowing sanitizer wall time; no ASan/UBSan diagnostic | executable `e3176533...`; selected log plus direct 3D restart result |

The Release full-suite log records the exact final result. The Debug full run
timed out only three long processes; the VC-specific 3D history/replay test
then passed alone. The O4 convergence assertion passed in Release with
component orders 3.9935--4.0009. The generic Debug AMR-history integration was
not extended beyond 600 seconds because it is unrelated shared infrastructure
and passed in Release.

## Restart, output, history, and replay

- Host VC restart tests pass for 2D Cartesian, 2D Cartoon, and 3D Cartesian,
  including pre-refinement, post-refinement, and post-derefinement
  continuation plus malformed/unsupported rejection.
- Host VC output tests pass in all three supported geometries. Nodal metadata,
  logical bounds, arrays, and the CC-adapter payload are checked separately.
- Ring-volume history quadrature is exact across Cartoon refine/derefine.
- Record/replay tests pass in 2D and 3D, including physical-time scheduling,
  exact logical-leaf sets, and restart continuation.
- Exact-final SYCL 2D restart and all three output cases pass. Exact-final
  SYCL 3D restart does **not** reproduce the final payload bit-for-bit; see
  `BACKEND_STATUS.md`.

## Preserved failing numerical discriminator

The nonconstant O4 dynamic-AMR wave test is intentionally not a registered
green CTest. It uses the same physical refine/derefine schedule at N16 and
N32 and compares the accepted final field:

| Geometry | N16 RMS | N32 RMS | observed order | disposition |
|---|---:|---:|---:|---|
| 2D Cartesian | `1.185856836e-6` | `2.289381583e-6` | `-0.9490280951` | fail |
| 3D Cartesian | `1.676664824e-7` | `2.283913019e-7` | `-0.4459133945` | fail |

The N64 follow-up after a speculative authority synchronization also grew
relative to N32; the experiment was reverted. No safe production correction
was established. The failure logs are retained under `evidence/local/`.

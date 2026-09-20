# Active-only characteristic boundary derivative review

Read-only review of `src/z4c/z4c_boundary_stencil.hpp` and its uses in `src/z4c/z4c_Sbc.cpp`, 2026-09-20. This covers the focused unavailable-RHS-ghost repair, not approval of the separately gated experimental physical-constraint boundary source.

## Findings

- At a local lower active edge the helper evaluates `(-3 f0 + 4 f1 - f2)/(2 dx)`; at an upper edge it evaluates `(3 f0 - 4 f1 + f2)/(2 dx)`. Both are second-order one-sided derivatives. All other cells use the original centered second-order derivative.
- The local active bounds, rather than incident physical-face flags, select the stencil. This is necessary because an inverse full metric with off-diagonal entries raises a physical-face covector to a vector with tangential components. The original tangent-centered stencil could then reach an internal-block RHS ghost that was never computed or exchanged.
- All configuration-field uses in the characteristic state and its rate now pass the same active indices: conformal metric, chi, lapse and shift. The field projections and the raised normal are unchanged. Physical `side` flags still define the physical frame and face ownership; they no longer incorrectly assert that all other directions have valid RHS ghosts.
- The derivative accumulation order, tolerance for negligible normal components and arithmetic expressions are unchanged on previously active-only stencils. In particular, an exactly diagonal metric normal retains its old path, including at tangent block edges. No blanket bitwise claim is made for skew normals at the repaired locations.
- The CPBC updates only momentum-type RHS variables Khat/Theta/A/Gamma. It differentiates configuration-type RHS variables chi/metric/lapse/shift, which remain immutable in these kernels. Face ownership is disjoint. The fix needs neither an RHS exchange nor additional synchronization.
- Three active cells are required for the one-sided formula. Production is three-dimensional and has at least four active cells per block direction by mesh validation. A reduced-dimensional off-diagonal use is outside this review and the proposed regression; no new claim of such support is made.

No blocker was found for the focused three-dimensional repair. It fixes invalid stencil inputs, not the separate continuum boundary modes or all possible long-time instabilities.

## Existing compiled unit coverage

`tst/unit/z4c_boundary_stencil/point_tests.cpp` poisons ghost cells, supplies independent quadratic state/RHS data, uses a determinant-one positive-definite metric with all three off-diagonal entries, and covers physical faces, edges and corners together with internal tangent edges. It checks finite new derivatives, an old-stencil nonfinite witness, exact zero, nonzero response, polynomial accuracy and linearity of the state/RHS derivative. This tests the actual new C++ helper on a host-accessible Kokkos build. It is not a GPU or full MPI evolution test.

## Implemented MPI integration regression

`tst/regression/z4c_boundary_oblique.py` uses the existing `z4c_tov_ks` characteristic-pulse initializer, so no new physics initializer is needed. Starting from `inputs/tests/z4c_characteristic_cpbc_plane_pulse.athinput`, it makes a three-dimensional box spanning [-1,1]M with 16 cells per direction and 8-cell blocks, outflow on all six faces and three RK3 cycles. It uses the existing oblique lapse pulse with `characteristic_test_oblique=true`, three oblique dimensions, amplitude 0.01, center 0.75, width 0.5 and transverse width 1. The background remains Minkowski, with `zero_rate` CPBC and no matter feedback. The oblique tensor supplies gxy/gxz/gyz and transverse variation while retaining a valid metric.

1. Save full state and pre/post-boundary RHS at every RK stage. Select actual physical-face cells that also lie at a local tangential low/high edge shared with another block. Assert that the inverse metric gives a contributing tangential raised-normal component there; otherwise the test did not trigger the bug.
2. Independently reconstruct the active-only derivative and full metric frame, and verify the incoming scalar/vector/tensor characteristic-rate equations after CPBC. Require finite active data, valid metrics, a nonzero boundary update and nine tested RK stages. Checking only finiteness could miss a finite but uncomputed ghost value.
3. Run exactly the same eight-block layout with one, two and four MPI ranks. Compare by global block identifier and stage, not rank-local order. With no reduction-dependent evolution in this vacuum control, the active stage arrays should be bitwise identical. This checks partitioning and ghost-exchange completion without changing the stencil layout.
The implemented script performs checks 1–3, including all-active metric positive-definiteness/lapse/chi checks and bitwise immutability of the configuration RHS during CPBC. Separate zero/diagonal controls cover unaffected-path preservation; this oblique script does not claim to run them. If block sizes are varied, assess accuracy by spatial convergence, not bitwise equality: the repaired tangent derivative is intentionally one-sided at local edges, so merging blocks changes that discrete stencil for nonpolynomial data.

The local MPI execution passed on 1, 2 and 4 ranks, then passed again using the final immutable executable with SHA256 `67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8`. Each run provided 144 sampled physical-face/internal-edge intersections over nine RK stages. Tangential raised-normal magnitudes ranged from 9.46e-6 to 0.004091, so the previously hidden path was exercised. The largest independent ten-mode rate error was 1.388e-17; the logged enforcement error was 9.324e-18. Boundary RHS corrections were nonzero (maximum 0.016271). All 288 active stage/block arrays and all 72 complete post-RK residual block payloads matched bitwise across rank counts. Final raw records are retained under the task's `stencil-fix/oblique-mpi-final` directory, with hashes and per-run summaries in `results.json`; the earlier run is retained separately in `oblique-mpi`.

Reproduce with a current MPI build, NumPy and an MPI launcher:

```sh
python3 tst/regression/z4c_boundary_oblique.py --exe /path/to/athena --output /new/output/directory
```

Use `--analyze-only` to recheck existing output. The default launcher is `mpiexec`; `--launcher` accepts a replacement command and flags. This is a short finite-pulse integration test, not long-time stability or GPU validation.

The old `z4c_standard_cpbc.py` regression uses a diagonal plane pulse and samples tangent midpoints. Its successful MPI check cannot detect this defect. A NaN-ghost full-kernel harness would be an additional strong witness if added later, but the compiled helper poison test already supplies a direct unavailable-input demonstration.

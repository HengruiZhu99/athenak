# Full scri consistency audit

This tests the current SG1 mesh and actual device execution kernels, on the local serial Kokkos backend. It does not evolve a black hole. The baseline uses the continuous Cartesian core, spherical overlap, equiangular shells, seven radial nodes, sampled interior pole numerators, original penalties, cross-face interpolation, angular epsilon=2 and physical-variable KO. Manufactured states are evaluated directly at every volume and surface point.

Three fixtures are used:

1. Smooth radial perturbations proportional to `(1-r*r)^2 sin(2.3*r*r+.37)`. These satisfy all tested scri pole/regularity conditions and have a nonzero analytic surface RHS.
2. The same regular perturbations with angular modulation and a nonconstant tangential B field on the surface. This also tests the angular filter and complete flux-field RHS.
3. A radius- and volume-preserving angular coordinate transformation of flat space, `y=Rot_z(.15*z/r)x`. Its nonconstant metric is `g=J^T J`, chi=1, A=0, K=-3 and Lambda=contracted Gamma. Third-order analytic coordinate jets supply the metric through second derivatives and Lambda through first derivatives. The geometric RHS is analytically stationary, but the stored fields, connections and their angular derivatives are not constant. The auxiliary D bulk RHS need not vanish and is included in the reference. This fixture complements the nonzero-RHS waves; it is not the exactly preserved constant Minkowski state.

All first jets and needed diagonal second jets are checked against independent finite differences. The analytic normal-pole AD reference is separately checked by five-point differences of exact pole numerators. Analytic pole and regularity residuals are required near roundoff. The complete surface output is checked against its reconstructed regular/normal/penalty/filter decomposition; all 31 output channels and the dependent-D trace are measured. The closest six volume rows include the actual transport correction and KO; the central geometry and pole/Omega errors are also separated. Norms use exact equiangular-cell solid angles. Global, two-cell seam-band, and face-interior errors, component maxima and their coordinates are saved. The seam band shrinks under refinement.

## Baseline finding

The regular wave fixtures approach fourth order in the full surface RHS, and approximately fifth order in the closest three volume rows. This alone would miss a defect. The nonconstant surface metric instead shows approximately third order in both the full surface RHS and closest volume rows. The surface error is dominated by the explicit regularity penalty; the volume error is dominated by pole/Omega. The sampled normal derivative alone is approximately fourth order in that fixture. Thus a genuine derivative-error amplification is measured, rather than inferred just from the explicit factors of 1/h.

The established baseline reports are `baseline-n12/16/24/32.json`; `baseline-binding.json` records source, binary and compile-flag hashes. Earlier `n*.json` (two fixtures) and `twist-n*.json` (three fixtures before the added independent jet check) are pilot records, not substituted for the final baseline. The analysis script uses only the named baseline and separate first6 control records.

## Reproduce

Configure the existing fresh `build-native-shared-transition` as documented in the initial-data audit. Build `hyperbolic_shift_3d_scri_full_consistency` and run it with `N fresh_output.json`, for N=12,16,24,32. It refuses existing output paths and never writes an evolution checkpoint. Run `python3 docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/checks/scri-full/analyze.py` for the summary and figure.

The measured defect requires a separately tested numerical repair before the controlled long evolutions can be presented as a convergence investigation. Passing this finite family of tests is not proof for all regular data or of stability.

## Separate first-derivative repair

`HYPERBOLIC_SCRI_FIRST6=1` changes centered first derivatives in the outermost spherical shell from FD4 to FD6 and changes the angular surface/last-three-row reconstruction from five to seven nodes. Seven radial nodes and six-point donor interpolation remain. The bulk second-derivative stencil away from the special radial rows remains FD4; transport remains the existing biased FD4 after its correction. The scientific equations, penalties, angular dissipation, gauge, floors and time step are unchanged. Both CPU and device sampled-pole kernels use the same modified derivative. This is separately selectable from KO10 and shared restoration. Original targets keep their original stencils.

The first-derivative accuracy supplies headroom for the measured amplification: exact smooth samples give sixth-order first derivatives away from seams; six-point cross-face value interpolation can give fifth-order first derivatives there. Dividing their error by h permits fourth-order truncation at seams. This is an accuracy argument for smooth regular data, not a stability proof. Device selection currently uses the outer spherical radius; these targets use the documented three-shell geometry.

Actual full-mesh reports `first6-n12/16/24/32/48/64.json` retain every component maximum and its location, and the global, seam and interior norms. At N32 the nonconstant-metric surface RMS error falls from .004301573 to .0000897293 (47.94 times smaller); nearest-row RMS falls from .001423962 to .0000293018 (48.60 times smaller). N48→64 effective orders are:

| Fixture | Full surface RMS | Sampled normal RMS | Nearest volume row RMS |
|---|---:|---:|---:|
| Radial regular wave | 5.031 | 5.022 | 5.057 |
| Angular regular wave + B | 5.024 | 5.012 | 5.033 |
| Nonconstant angular metric | 4.722 | 5.671 | 4.730 |

The angular-metric seam RMS orders are 4.216/4.224 for surface/nearest row. The **pointwise seam maxima remain pre-asymptotic**: surface order increases 3.760→3.819 on the last two pairs, and nearest-row order 3.762→3.821; row3 is 3.706 and row6 3.769 on the last pair. The maximum is near a face corner. Wave rows4–6 retain FD4 bulk terms, with global RMS orders approaching4 (row6 about3.984); their finite-grid maximum orders are still below4. We do not claim every pointwise effective order is at least4 or that this finite test family proves generic consistency. The proven third-order global defect is removed in these tests; remaining maxima and late evolution must remain visible.

`parity-tests.log` records passing original spherical-overlap and new first6 CPU/device tests, including sampled numerator, surface closure and partition checks. `smoke-results.json` binds two-step actual native MPI runs for Minkowski and puncture data with shared restoration, KO10 and first6 together. Both completed and admitted distinct NS2 checkpoints, with finite geometry and roundoff null closure. They establish plumbing, not long-time stability. `first6-binding.json` records tested source, binary and compiler flag hashes. New production schema IDs NC1/NS1 and NC2/NS2 distinguish original/shared collar and KO8/KO10 with first6; prior checkpoint IDs are unchanged.

Reproduce the new manufactured runs with target `hyperbolic_shift_3d_scri_first6_consistency`, and the parity test with `ctest --test-dir build-native-shared-transition -R 'hyperbolic_shift_3d_(spherical_overlap|scri_first6)_device$' --output-on-failure`. The smoke commands and checkpoint hashes are in their JSON record. No SYCL or long evolution result is claimed yet.

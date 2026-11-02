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

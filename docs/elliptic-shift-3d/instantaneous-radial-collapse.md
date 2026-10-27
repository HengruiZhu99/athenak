# Frozen radial collapse: instantaneous composite 3D shift comparison

The three selected, completed N4097 spherical snapshots at t=1,5,10 were
sampled onto a genuine composite 3D grid and used for nine global vector
elliptic solves. All solves reached the original-equation algebraic tolerance.
Their spatial agreement remains provisional: this N6 grid produces shift
errors of order1e-3 with independent interpolated physical sources and order
1e-2 with the pilot's differentiated state. This is an instantaneous elliptic
comparison, **not a 3D evolution, spatial convergence result, puncture endpoint
validation, or completion of the scientific milestone**.

## Equation, grid and independent source control

`instantaneous_radial_snapshot.cpp` includes the existing coupled research
pilot with its main renamed; it does not modify that pilot or native AthenaK
execution. The mesh has an AMR-capable Cartesian interior subdivided into
2x2x2 blocks and three six-face shell layers ending at .62,.70,1, for26 blocks
of6x6x6 active cells,5616 cells total. The Cartesian halfwidth is .25. Existing
fourth-order mapped differentiation, radial interface stencils, composite
communication, original-vector residual evaluation and multigrid-preconditioned
solve are used. The solve connects every block globally. No block is solved
with permanently fixed neighbor data.

The state stores log chi and log(alpha/Aref), where Aref=(1+r²)/2, using the
fixed exponent-independent positive chart. Signed Atilde/Khat/Theta are direct
physical fields, Khat's stored slot includes the existing+3 offset, independent
Lambda and telegraph B are physical Cartesian components, and the recorded
spherical conformal metric is exactly delta_ij. The historical source factors
are used only to deserialize each snapshot. No current power assumption is
made. The unchanged Minkowski-reference equation is solved with eta V=0 for
this exactly flat conformal metric. The complete curvature source is evaluated
from the physical alpha and A tensor; no `L(beta_recorded)` substitute is
inserted into any RHS. The recorded shift is a warm start and comparison only.

Each frozen snapshot supplies three distinct sources:

1. `FD`: differentiate the sampled pilot state using its actual 3D ghost/interface
   machinery, then evaluate the complete original source.
2. `interpolated7`: evaluate that same tensor source from exported Cartesian
   value/gradient/Hessian jets, independently interpolated from7 radial nodes.
3. `interpolated9`: repeat the export with9 radial nodes.

The latter two are interpolated numerical reference jets, not exact analytic
solutions. The recorded beta's independently differentiated continuum residual
is retained as a reference consistency check. Alpha*A is the original source
in all three cases. The affine reference beta=-x is handled by the pilot's
existing shifted unknown; its exact operator contribution vanishes for this
flat metric and is explicitly subtracted. There is no puncture beta pin.

The r=1 boundary uses the actual frozen chi and prescribed lapse clock alpha=1.
Existing future-null normal data and fixed reference angular coordinates supply
all three shift boundary conditions. Their ghost-reconstructed normal and
tangential residuals are checked separately. No finite-radius outer boundary,
Omega floor, potential boundary shortcut or scri multipole condition is used.

## Frozen repeat and preserved first attempt

The initial attempt, in
`evidence/instantaneous-radial-collapse-20261027`, finished all three cases but
correctly flagged concurrent source edits to coupled.cpp and
puncture_log_plans.hpp. Its executed binary was stable. That unqualified source
provenance is preserved, rather than overwritten.

A separate repeat in
`evidence/instantaneous-radial-collapse-frozen-20261027/results` compiled the
archived driver and archived coupled.cpp against the archived source include
tree. Its409 source files were captured before rebuild in `source.tar.gz`.
All835 hashed Kokkos headers/static libraries were unchanged over the repeat;
header copies are additionally retained in `runtime-headers.tar.gz`. The
executed binary SHA256 is
`a9ad2d583cae49b0944af06127d85058eb4307605ea0ddc85f4c801813fdf742`.
Source and binary stability flags both pass, and the numerical results match
the first attempt. `reproduction.json` records one precise metadata limitation:
the generated AthenaK config was copied after the run, with its earlier mtime,
rather than carrying a prebuild config content hash. The Kokkos dependency
hashes were captured before and checked after execution.

Each case had a180-second timeout; actual complete solve-process durations
were9.58,9.65,9.64 seconds. This process performs no timesteps. Per-solve cost
was2.5–2.9 seconds, with51–58 iterations. These small Serial Kokkos CPU tests
establish no scalable GPU/MPI performance result.

The recorded-beta warm start has original FD-equation physical RMS residuals
.677,1.83,1.70 at t=1,5,10; the converged residuals below demonstrate roughly
10–11 orders of algebraic reduction. The independent-source initial residuals
are .788,1.22,1.11. Initial and final residuals are both evaluated with the
original discrete vector operator and its boundary data.

| Frozen time | FD-source beta RMS | Independent-source beta RMS | Width9-minus7 beta RMS | Maximum original residual RMS |
|---:|---:|---:|---:|---:|
|1|6.02281e-3|9.99972e-4|3.82253e-11|5.89591e-11|
|5|6.43432e-3|9.91498e-4|6.27396e-13|5.49389e-11|
|10|6.62082e-3|1.00082e-3|6.49224e-13|5.63540e-11|

RMS uses Cartesian vector component squares weighted by physical spatial
volume. Conformal coordinate-volume RMS, maximum component/vector errors and
all regional splits are also retained. Every active cell remains in the
all-region norms; the overlapping core, Cartesian, shell, interface-neighbor,
edge/corner-neighbor, transition and scri masks are diagnostics, not acceptance
exclusions. Normal boundary residuals are below8.9e-16 and tangential below
3.3e-16. All sampled active chi/alpha values are finite and positive.

The independent recorded-beta continuum residual RMS is1.89e-7,2.62e-8,2.73e-8,
and the corresponding source width9-minus7 RMS is1.30e-8,6.61e-10,5.96e-10.
Those controls are far smaller than the coarse-grid discrepancies. Decreasing
the linear solve tolerance would not fix the observed spatial error.

## Localized failure mechanisms and unresolved regions

The FD-minus-independent source RMS is.737,2.79,2.52 over all active cells.
Within r<.1 it is.836,9.97,9.23; within .62<=r<=.70 it is1.20,.703,.737. At t1
its largest vector error2.865 occurs at r=.68. At t5/t10 maxima12.70/11.84
occur at the innermost sampled radius .0360844. These indicate two separate
spatial controls: transition/source differentiation and late inner gradient
aliasing. They do not establish a mathematical failure of the formulation.

The FD-source shift error is predominantly radial. Its maximum vector error
.01191,.01191,.01224 occurs at r=.693333, near the outer edge of the explicit
transition layer. Using the independent source lowers its maximum to
.00235,.00218,.00223, now at r=.725 in the first cells of the outer layer.
The remaining discrepancy includes the discrete vector operator and interface
transfer; an additional independent operator/transfer control is required to
separate those. No single-resolution conclusion of formal order is made.

The minimum active radius is .0360844. The actual late-time B/front features
around1e-7 are vastly smaller. No sample at r=0 was requested; no limiting data,
regular-center parity or physical endpoint division was invented. Therefore
these tests do **not** validate the changing AF/trumpet endpoint or its late
front. Unresolved positive-radius data remain in the full norms. N6 happened
to sample no zero/nonpositive point, so an endpoint error was not exercised.

`angular-comparison.png` shows means and raw scatter over64 independent
Cartesian directions at13 radii for each snapshot. The identical extraction
of the recorded radial target supplies a projection-error control. Raw scatter
can contain that extraction error, and is not automatically a physical
nonspherical mode. The 3D solve was never angular averaged or symmetry
projected; angular averages are diagnostics only.

## Reproduction and next gates

The reusable driver, runner and analyzer are in `tst/elliptic_shift_3d`.
For a fresh output directory after extracting the source archive:

```sh
mkdir -p build-shift-3d/new-frozen-snapshot-source
# Extract the archived source.tar.gz into that directory, preserving paths.
python3 tst/elliptic_shift_3d/run_instantaneous_radial_snapshots.py \
  --frozen-source build-shift-3d/new-frozen-snapshot-source \
  --output build-shift-3d/new-instantaneous-radial-results
python3 tst/elliptic_shift_3d/analyze_instantaneous_radial_snapshots.py \
  build-shift-3d/new-instantaneous-radial-results --plot
```

The runner uses the local Serial pilot's existing CMake compile/link settings,
which are copied with dependency hashes in the evidence directory. Snapshots
and all raw Cartesian inputs/outputs are retained losslessly; compressed-file
manifests give their original byte counts and hashes. `analysis.json` adds exact
error-maximum coordinates and all regional/angular comparisons. No original
spherical file or job was changed.

The next scientific gates are derivative/product-source and vector-operator
controls at the localized transition cells, a related-resolution study,
resolved changing puncture endpoint treatment, and genuine coupled 3D
evolution/restart/regridding tests. These instantaneous results provide a
falsifiable starting comparison, not a substitute for those gates.

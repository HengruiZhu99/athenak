# Pinned GPU comparison assumptions

This read-only audit found no numerical-setting or boundary-formula mismatch
in the inspected paths. It does not replace the matched original/native GPU
trajectory comparison. The earlier CPU oracle compared native Driver tasks
against the old CPU `SphericalTransfer3D` path, so that test alone cannot prove
parity with the historical GPU executable.

The historical `evolve-static` manifest enables `HYPERBOLIC_SPHERICAL_POLICY=1`.
The exact pinned `coupled_mpi.cpp` (SHA256
`87a8d90396bb3ab6e72efadab0e9491322517bf63413f8f4460a42f24f51c115`)
**does not use compensated acceptance for this policy**. Its bulk branch at
lines 1286–1288 and scri branch at 1357–1359 evaluate
`base + dt*(k1 + 2*k2 + 2*k3 + k4)/6`, assign the next compensation to zero and
return before the generic compensated update. Native `ClassicalRK4::Advance`
accumulates that same left-associated unscaled numerator and applies the same
base update. Intermediate stages use the same `base + (dt/2 or dt)*rhs`.
The historical restart's compensation arrays are inherited storage, not an
active missing state variable for N12.

Both builds record the same icpx path and numerical/device settings:
`-fp-model=precise -O2 -ffp-contract=off`, SYCL target `spir64_gen` with device
`12.60.7`. This was checked for native RHS, surface, update, reference-cache
and chart-atlas translation units. Original `gnu++17`/PIE versus native
`c++17` and kernel/translation-unit layout remain build differences; equal
flags do not guarantee identical generated device arithmetic.

The inspected pinned surface implementation and frozen native code preserve:

- Algebraic chart expansion, null-speed projection, lapse-offset zero, J zero,
  and the five-point outward extrapolation of `D-Lambda` from the owning radial
  block, in the same order.
- One base boundary evaluation and three unit shift-gradient probes for the
  normal-compatibility matrix, followed by the same partial-pivot Gaussian
  elimination. Native additionally rejects NaN pivots; finite valid matrices
  retain the same `1e-12` threshold and arithmetic.
- Six interior radial nodes plus the actual evolved endpoint, centered 5×5
  angular stencils, and six-point cross-face interpolation, preserving donor
  insertion and summation order. Endpoint reconstruction covers the last
  three bulk radial rows.
- The same radial polynomial normal lift and local 6×6 angular interpolation,
  applied to scri and only the last two bulk radial rows.
- The same independent-field and second-derivative selection, compatible
  lapse-Hessian correction, reference restoration followed by normal lift,
  angular-only endpoint transport correction and chi `r*r` transport factor.

Normalized function bodies for `Physical`, `Restore`, `Boundary`,
`CompatibleWeight`, central parameter resolution and the reviewed wormhole
quadratures match. Both boundary sources are `reference.source.At(1)`; bulk
sources are evaluated at the mapped radius. The inputs preserve 20 M central
radius, .97 slicing endpoint, constant shift-speed policy, .25 outer lapse
damping, .5 scri penalty, .1 KO, `1e-10` chi floor and the fixed timestep.

The original snapshot calls `RHS` before writing. `CompleteState` itself only
gathers. The RHS refresh repeats dependent scri projection, which accepted
states already undergo; it does not advance independent fields or chi history.
The stored 31-field conventions are shared Cartesian components in both
formats. The independent canonical map covers all 229,392 evolved bulk points,
including evolved overlap, and all 864 scri points. Historical output only
provides an aggregate chi ledger, so per-cell record equivalence must not be
invented from it.

Exact source checks, compile settings, reviewed paths and findings are recorded
in [evidence/native-gpu-comparison-assumptions.json](evidence/native-gpu-comparison-assumptions.json).
All inspected current native files match the frozen Aurora source snapshot.
No runtime code, tolerance or scheduler job was changed by this audit.

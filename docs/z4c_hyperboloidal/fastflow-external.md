# Native FastFlow external geometry extension

The original `FastFlow` class now accepts scalar or batched physical Cartesian
ADM first jets. The native mapped sampler supplies the batch callback, so the
finder retains its existing angular grid, harmonic update, geometric integrals,
and output machinery without allocating an ADM volume or gathering mesh fields.
Only surface samples and scalar/harmonic reductions are communicated.

The extension checks exact unique ownership and valid geometry collectively on
every trial. Invalid trial radii are rejected before a communicating callback.
Incomplete surfaces, callback exceptions, and invalid owned jets fail coherently.
An external output-file failure is also propagated collectively. `GetResult()`
provides an explicit current-search status; failed and skipped searches cannot
return a previous accepted horizon. `GetSurface()` copies the complete real
harmonic coefficients and search center for the exterior constraint mask.

The external path includes every angular quadrature sample, matching the
predecessor hyperboloidal extension. The original Cartesian path's exclusive
upper bounds omit the final point in its minimum-radius and surface-integral
reductions. Those bounds remain unchanged here; correcting them would be a
separate numerical change.

## Validation

The actual native FastFlow object passed all 15 required gates on two and eight
MPI ranks, with one identical test executable and unchanged source/object
manifests. Eight ranks include six ranks owning no analytic surface queries.

- Isotropic Schwarzschild gives area `50.265482457436768`, mass
  `1.0000000000000007`, and expansion RMS `2.8427561731765954e-16`.
  Scalar and batch results are bitwise equal.
- Starting at radius 0.65 converges in 137 iterations to
  `0.50000000099651087`, with expansion RMS `9.9651063639409561e-10`.
- Missing/duplicate ownership, invalid flags/metric, rank-local callback errors,
  and a rank-local invalid radius all fail collectively. Failed-after-success,
  skipped, iteration-limited, and output-open-failure cases pass their gates.
- The translated nonspherical l=2 fixture checks all real harmonic components
  against independent closed forms and classifies 3840 near-surface points.
- The original Cartesian puncture problem runs for two steps on both the saved
  prechange Athena executable and the final executable. Horizon summary and
  shape files are byte-identical before/after separately on two and eight ranks.

The original executable and FastFlow/GaussLegendre headers, sources, and objects
were archived before rebuilding. Reproducible commands and detailed assertions
are in `tst/z4c_hyperboloidal/fastflow_external.md`; durable hashes and results are
in `evidence/native-fastflow-validation.json`. These tests establish correctness
and compatibility, not performance or the native mapped sampler's accuracy;
the latter has its own predecessor oracle.

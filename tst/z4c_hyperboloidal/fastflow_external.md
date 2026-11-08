# FastFlow external geometry contract

The native diagnostics integration extends the existing `FastFlow` solver with
physical Cartesian ADM first jets. It preserves the existing harmonic basis,
flow update, geometric integrals, output format, and ordinary Cartesian path.
It does not allocate ADM volume storage for an external sampler. The associated
`GaussLegendreGrid` change supplies angular quadrature without constructing
Cartesian interpolation metadata.

The scalar callback returns true exactly on the owning MPI rank. The optional
batch callback supplies a geometry and 0/1 ownership flag for every query. A
batch callback may perform collective sparse transport; it must make its own
rank-local errors collective before returning or throwing. Scalar callbacks
must not perform collectives. All FastFlow ranks participate even when they own
no samples. Every trial requires exactly one owner and a valid positive-definite
finite geometry at each angular sample. Invalid radii are checked collectively
before entering a communicating callback. Missing/duplicate ownership, invalid
owned geometry, and callback exceptions fail the search collectively.

`GetResult()` distinguishes not-searched, converged, incomplete-surface, and
other failed searches. Accepted quantities and `GetSurface()` are unavailable
for a skipped or unsuccessful current search; separate last-trial quantities
remain available for diagnosing nonconvergence. `GetSurface()` carries all real
harmonic coefficients and the actual search center. Its exterior mask evaluates
the complete shape, with conservative harmonic bounds as an acceleration.

External geometry uses every Gauss-Legendre quadrature point, matching the
predecessor hyperboloidal extension. The original Cartesian code's exclusive
upper bounds omit its final quadrature sample in two reductions. This extension
retains that existing Cartesian behavior deliberately; it does not claim to fix
it. A separate numerical change would need its own admission and regression.

## Gates

`fastflow_external.py` links the actual built native FastFlow object and runs:

- Isotropic Schwarzschild, with analytic area `16 pi`, mass 1, coordinate radius
  1/2, and zero outgoing expansion. Scalar and batch results must be bitwise equal.
- A nontrivial flow starting at radius 0.65, converging with expansion RMS <=1e-9.
- Missing/duplicate ownership, invalid owned metric, invalid ownership flags,
  rank-local scalar and batch exceptions, and rank-local negative trial radius.
- Failed-after-success and skipped searches without stale accepted quantities;
  finite last-trial diagnostics for an iteration-limited nonconverged search.
- A rank-local output-open failure propagated to all ranks.
- A translated surface containing l=2,m=0 and both l=2,m=2 real modes, checked
  against closed-form harmonics at 1920 angles and 3840 inside/outside points.
- The existing Cartesian puncture case, run for two steps using both the archived
  prechange executable and the final executable. Summary and shape output must
  be byte-identical, finite, and nonempty.

Both two-rank and eight-rank executions use the same external-test executable.
The eight-rank analytic case has six observed zero-owner ranks. The runner
requires every named gate and records executable, runtime source, and linked
object SHA256 manifests. A correctness run supplies no performance evidence.

Example (run only after a coordinated native build):

```sh
python3 tst/z4c_hyperboloidal/fastflow_external.py --build build-native \
  --output build-native/fastflow-external-two --ranks 2
python3 tst/z4c_hyperboloidal/fastflow_external.py --build build-native \
  --output build-native/fastflow-external-eight --ranks 8 \
  --executable build-native/fastflow-external-two/fastflow_external
```

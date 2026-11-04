# Terminal large-center separated-driver control

Job9038253, frozen scientific source342715a90, native/worker/PBS failure2.
Last accepted time0.463875M,1237steps; last diagnostic0.45M. Native command
elapsed342.518s includes initialization. All151 worker artifacts are locally
hash/length verified. Matching lock released. No checkpoint is admitted and
no restart/replay is permitted.

Raw evidence remains under
`output/refined-convergence/frame-separation-aurora/campaigns/stationary20-separated-n16-segment01`.
`local-verification.json` binds each artifact and the archive. The raw failed
active array has1356640cells,31doubles per cell,336446720bytes. All values are
finite; all active chi and metrics are positive. Minimum metric eigenvalue
is0.2326642932. These facts alone do not guarantee positive interpolated halos.

`failed_halo_audit.inc`, built by `build_endpoint_diagnostic.py --body ...`,
reconstructs production non-boundary halo interpolation using the admitted
local two-step checkpoint only for its matching topology. It does not evolve
or admit the failed capture. The exact production donor weights and frame
transformations produce zero invalid rows out of3503736 checked. Surface-
dependent13824 rows are excluded. This is a CPU reconstruction, not bitwise
SYCL replay. Build commands/hashes are in `halo-build.json`; invocation:

```
mpiexec -n 2 output/refined-convergence/failed-halo-audit/reader \
  output/refined-convergence/separated-stationary-n16-mpi2.checkpoint \
  inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial12_separated.cfg \
  output/refined-convergence/frame-separation-aurora/campaigns/stationary20-separated-n16-segment01/output/state.failed-stage.bin \
  output/refined-convergence/failed-halo-audit/failure 16
```

The old AMR failure writer omitted `Evolution::failed_edge`, despite the RK
exception handler retaining it. Therefore this record cannot distinguish a
bad surface interpolant from a bad surface-dependent radial halo. The error
string also covers ordinary AMR halos; interpreting it by name alone was too
specific. The new local diagnostic change preserves surface data separately
and distinguishes error counts for future new builds. Frozen artifacts remain
unchanged. Constant-driver job9038285 is the fresh matched attribution control.

The horizon remained accepted at0.45M with mass0.33333348265 and expansion
RMS3.1783e-5. Nullness remained near roundoff while cumulative pole/clock maxima
grew to161.81/53.04. The fixedr>.6 errors already grew by0.15M. This does not
show near-puncture slicing breakdown or establish the sole instability cause.
`plot_separated_failure.py` generates the inspected history PNG/PDF. Curves
connect only four recorded rows; gauge maxima are full-domain, not exterior.

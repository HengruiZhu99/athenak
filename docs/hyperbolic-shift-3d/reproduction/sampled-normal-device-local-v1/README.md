# Optional sampled-normal device component

The new native device component and existing scri-device regression passed in
Kokkos **Serial**, in 4.87 s and 1.87 s respectively. This is a bounded component
check, with no evolution or GPU stability claim. The optional sampled-numerator
closure has the separately measured roughly third-order manufactured full-rate
accuracy described in `../scri-sampled-accuracy-v1/`; it is an experimental
alternative, not an admission of fourth-order accuracy or long-time stability.

Both n=6 fixtures (Minkowski and nonspherical perturbation, angular epsilon=2)
have zero recorded CPU/device error for all interior pole samples, all 31
boundary rates, and all 31 final surface fields. Twelve contiguous surface
ownership partitions, a range crossing a face boundary, and an empty range
agree exactly with the single full-range evaluation. Halos outside each owned
surface-block range were poisoned with NaNs before filling. The test also
constructs a separate Cartesian-core volume owner, so it does not rely on
volume ownership supplying the surface donors' halos.

Finite D perturbations in both the volume and dependent surface storage leave
all geometric numerator samples, rates and final surface fields bitwise equal.
Invalid shapes, missing/wrong-mode samples, forbidden sample channels,
nonfinite numerator samples and nonfinite native D donor values are rejected.
Maximum null residual is 1.1102e-15 and dependent-D compatibility residual is
2.4287e-17. See `results.log` for exact values.

The test compiles the frozen d98e2a99bbf1d5546956be3d724c82715ca704ec scri
device header alongside the current one, changing only its class name and
include guard. With sampled-normal mode disabled, the complete RHS and final
surface arrays match this historical kernel by raw-byte comparison, including
signed zeros. `receipt.json` records both the original header digest and the
test fixture digest; its transformation was independently checked during
packaging. This tests the same backend/compiler's default arithmetic. GPU
compilation and parity still require the Aurora preflight.

The enabled implementation samples five native active radial cells per owned
surface row after a dedicated provisional halo fill. It reuses the bulk
device jet construction and complete damped native pole, retaining the actual
interior Omega. The scri kernel combines those values with its current endpoint
pole in the CPU accumulation order, then performs the existing penalties,
angular source, null-clock and dependent-D reconstruction. The runner must
gather the final dependent D and refill volume halos before bulk evaluation.
The dedicated exchange has a global-shaped padded buffer and repeats the
active transpose; that experimental memory/copy cost is not hidden as a
performance improvement.

Reproduction from the repository root:

```sh
cmake -S . -B build-hyperbolic-mpi
cmake --build build-hyperbolic-mpi -j2 --target \
  hyperbolic_shift_3d_sampled_normal_device \
  hyperbolic_shift_3d_mpi_sampled_normal \
  hyperbolic_shift_3d_mpi_dissipative \
  hyperbolic_shift_3d_scri_device
ctest --test-dir build-hyperbolic-mpi \
  -R '^hyperbolic_shift_3d_(sampled_normal_device|scri_device)$' \
  --output-on-failure --timeout 390
```

This used the existing MPI-enabled, double-precision Serial Release build;
configure arguments are inherited from its cache. Exact source and binary
hashes, compiler flags and log hashes are recorded in `receipt.json`. The
coordinated build included the runner targets but this package's tests concern
only the device components; MPI/restart evidence is retained separately.

# Field-batched direct radial derivatives — local prototype

The V21/V22 ownership audit found that ranks with any direct radial closure rows
were expensive, even when only 69 such rows were owned. Reducing their cell count
did not reduce the slowest RHS time. See the [ownership evidence](../measured-partition-v1/README.md).
This motivates a kernel organization experiment; it does not establish a GPU
occupancy or compiler diagnosis.

`HYPERBOLIC_RADIAL_FIELD_BATCH=1` selects an experimental execution path. The
`hyperbolic_shift_3d_mpi_continuous_positive_radial_batch` target uses the existing
sparse exchange, equal-cell ownership and boundary reuse. Default targets retain
inline radial derivative application.

The radial plan still builds donor indices and derivative weights once. It now
optionally allocates a compact row lookup and scratch jets for selected rows.
`Prepare` refreshes derivatives in **one launch across selected rows × fields**,
using the same `Kernel::Derivatives` summation order. Bulk RHS, bulk diagnostics,
and sampled-normal numerator evaluations consume these current-stage jets.
They are recomputed on every call; no state-dependent derivative is treated as
mesh-only data. Scratch is reused only as storage. A returned view belongs to
its plan and is valid until the next preparation; this synchronous runner finishes
its device reduction before another preparation.

The generic batched pass computes first and second derivatives for all 31 fields;
consumers requesting first derivatives alone receive zero second derivatives,
as before. This adds work for unused second derivatives and memory traffic, so a
GPU speedup must be measured, not assumed. Storage is 12 doubles per selected
row/field plus integer row lookup/IDs. No interpolation order, donor location,
physical equation, floor, characteristic criterion, null closure, or RK sum is
changed. The V11 checkpoint schema is unchanged. Reports identify the path as
`radial_derivative_execution: stage_field_batch`.

## Local validation

- The derivative test passed **110,019 field-row comparisons** with zero error
  against CPU and exact value equality against direct device application. It
  reuses each plan for three changed input states, toggles first/second derivative
  requests, and covers twelve simulated partitions, noncontiguous and empty
  ownership, poisoned radial ghosts and invalid shape/index rejection.
- All **eleven** poisoned-workspace MPI cases passed. Complete pulse checkpoints
  match with one/two ranks and a one-to-two-rank restart; Minkowski and puncture
  checkpoints match the existing boundary-reuse control byte for byte.
- A rejected RK stage preserves exactly the same rollback checkpoint and complete
  failed-stage capture. Scientific acceptance and zero lapse-floor interventions
  are checked by the unchanged regression runner.

Reproduce the build and bounded tests:

```sh
cmake -S . -B build-hyperbolic-mpi
cmake --build build-hyperbolic-mpi -j2 --target hyperbolic_shift_3d_radial_field_batch hyperbolic_shift_3d_mpi_continuous_positive_radial_batch
build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_radial_field_batch
python3 tst/hyperbolic_shift_3d/test_sparse_stage_mpi.py \
 --binary build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_continuous_positive_radial_batch \
 --control build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_continuous_positive_fast \
 --mpiexec /opt/homebrew/bin/mpiexec \
 --initial output/hyperbolic-core-shell/continuous-positive-local-v1/puncture-n16.txt \
 --output-dir output/hyperbolic-core-shell/radial-field-batch-local-v1/mpi
```

Use a fresh output directory. The MPI runner has a 410-second aggregate deadline;
individual local tests are bounded by 420 seconds. Small logs, reports and binary
hashes are retained in `local/`; large state/failure files remain in local output.
This is **local fidelity evidence only**. SYCL fidelity, same-node performance,
and full GPU failure/restart qualification remain pending. It is separate from
the already frozen V23 timer-only job 8987400. Do not promote this prototype or
infer long-time stability, convergence or AMR evolution from these checks.

The broader native operator target `hyperbolic_shift_3d_radial_batch_device` also
passed the existing Minkowski/nonspherical parity matrix: bulk RHS, complete scri
rates, normal samples, twelve-way ownership, poisoned halos and invalid-input
rejection. These checks used the Serial backend locally and do not stand in for
SYCL qualification. Bulk scaled error against the CPU reference was at most
4.33e-19; sampled-normal and scri rate errors were zero.

The Aurora controller now treats the candidate as its own V11 execution identity
and requires `radial_derivative_execution=stage_field_batch` in admission. The
new `radial_field_batch` component adapter requires a SYCL backend, exact direct
derivative equality, all 110,019 field-row comparisons, three changed-stage
refreshes, and CPU scaled error <=2e-12 (the native derivative test tolerance).
All 105 offline controller/worker tests passed; the final admission and parser
changes were checked again with two focused tests. No GPU admission was issued.

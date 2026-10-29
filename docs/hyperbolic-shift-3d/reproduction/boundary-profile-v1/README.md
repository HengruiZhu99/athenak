# Boundary timing instrumentation

Adds observability only to `coupled_mpi.cpp`; no equations, interpolation sums,
RK stages, rejection paths, or checkpoint fields change. This instrumentation
is **not** in frozen source-v22 used by Aurora job 8987308.

`boundary_phase_seconds_by_rank` has six columns: null projection, sampled-normal
halo fill, sampled-normal numerator RHS, angular residual/application, scri RHS,
and boundary error/statistics MPI synchronization. These subdivide the existing
closure bucket; surface gathers remain in the existing communication bucket.
They do not include every minor closure operation, such as the projected surface
copy. Each timed operator already completes its device reductions/fence before
returning, so no additional device fence is inserted.

`error_sync_seconds_by_rank` measures all calls to `SynchronizeError`, including
startup, diagnostics, and RHS checks. It overlaps the sixth boundary column and
other existing phase accounting. **Do not add it to exclusive phase totals.**
MPI timings include waiting and are not network transfer time. Timers on a rejected
stage can be partial; use completed runs for performance comparisons.

Local validation used the balanced target built with `make -j2`, two MPI ranks,
poisoned unused sparse workspace, and the same 61/39 ownership fixture as the
previous local fidelity run:

```sh
HYPERBOLIC_SPARSE_POISON=1 mpiexec -n 2 \
 build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_continuous_positive_balanced \
 12 output/hyperbolic-core-shell/boundary-profile-local-v1/pulse \
 pulse .00075 .00025 - - 120 .001 .001 6 12 2 \
 output/hyperbolic-core-shell/sparse-stage-local-v1/balanced-bitwise/partition-test-n12-r2.txt
```

The complete 38,569,112-byte checkpoint equals the pre-instrumentation fixture
`output/hyperbolic-core-shell/sparse-stage-local-v1/balanced-bitwise/two.checkpoint`.
SHA256: `944ee7c7e04f423dbabf9b21c4bc9fbd3186788d726930ceb4c443b33d150f44`.
All timers are finite/nonnegative and each rank's six subphases sum to no more
than its aggregate closure time. Small raw evidence and verification are adjacent;
large checkpoints remain in local output. This is a fidelity check, not a GPU
performance claim.

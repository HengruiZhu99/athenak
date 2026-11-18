# Native multirate qualification

`qualify.py` launches the real AthenaK executable on a 25-block N12 Scheme A
fixture, writes native shared checkpoints, and checks accepted fields and all
eight persistent chi-record scalars. Each invocation gets an isolated results
directory containing commands, logs, binary/input hashes and a JSON report.

The gates exercise uniform and mixed rates, incompatible explicit levels,
same-schedule bitwise restart, noncontiguous stable-GID ownership, numerical
agreement across placements, and bitwise restart across a replayed redistribution.
The replay uses the runtime's actual graph fingerprint and moves blocks at a
common endpoint. The uniform/mixed tiny-step comparison is a smoke check; it
does not establish temporal order, pulse transmission or scri accuracy.

```sh
python3 tst/multirate_native/qualify.py \
  --binary /tmp/hyperboloidal-multirate-build/src/athena \
  --output /tmp/multirate-native-results \
  --launcher 'mpiexec -n {ranks}' --ranks 2
```

Enable CTest registration with `Athena_ENABLE_MULTIRATE_NATIVE_TESTS=ON`.
`--cases` selects an individual gate during diagnosis. A serial build skips
the MPI ownership and redistribution cases, so its report is not a substitute
for the MPI report.

## Performance runs

`benchmark.py` prepares commands for an existing one/two-node debug allocation;
it neither allocates resources nor submits jobs. Give it the launcher's explicit
GPU binding, available GPUs, several active GPU counts, and the conversion from
coordinate time to M0. Use both the current mesh and larger workload inputs.
Uniform reference uses the same base RK tableau at the finest local step.
Scalar and METIS cases cover the same coordinate-time interval. Existing output
configuration is retained, so select identical synchronized output schedules
and account for output overhead when interpreting elapsed time.

Without `--execute`, the script only writes reviewable `commands.json`.
An optional `--gate-report` records the status of numerical, communication, scientific and restart qualification alongside the timings. Production promotion requires separate review.

The runtime receipts and full logs are retained with wall seconds per M0 and
stage counts. Compare measured per-level work, batch-size profiles, communication
waits, peak retained-state memory and migration/rebuild overhead from those
receipts. CPU smoke results do not establish GPU scaling or speedup. Strong
scaling ends when communication and small batches dominate; sparse fast regions
are allowed to occupy a rank subset.

## Adaptive spatial refinement

With `<mesh_refinement>/refinement=adaptive`, `ncycle_check` and `refinement_interval`
count coarsest steps. An ordinary checkpoint at a refinement cut contains the new mesh.
The `multirate_adaptive_endpoint` test compares a real Cartesian refinement with
a fine uniform EX5 integration regridded at the same physical time and checks
bitwise restart across that refinement. An incompatible new edge or CFL requirement
stops evolution before the next interval.

On Aurora, `aurora/debug.pbs` uses two debug nodes and the CompactBinaryMerger
account. The runner checks each rank's tile binding and records one physical GPU
per two tiles. Steady timing uses the driver's wall clock between common cuts,
including intervening communication and load balancing. Initialization, final
checkpoint output and first regrid costs are also included in whole-process wall time.
`aurora/compare.py` checks shared checkpoints and computes strong-scaling efficiency
relative to the same workload and method on one physical GPU.

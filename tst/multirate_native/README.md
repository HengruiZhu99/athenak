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

## Selecting the stepper

Set `<time>/integrator=mrgark4` and set `<time>/dt` to the coarsest interval in
coordinate time. A level of 3 takes eight local steps per interval. Set
`<multirate>/spherical_levels=auto` or give a comma-separated integer list in
order of increasing physical radius. The startup log lists the resolved levels,
overlap promotions and limiting characteristic block.

| Multirate input | Default | Meaning |
| --- | --- | --- |
| `balance` | `metis` | Stage-aware placement. Build with `Athena_ENABLE_METIS=ON`. `scalar` selects the scalar-cost control and `none` keeps the starting partition. |
| `ranks_per_gpu` | 1 | Consecutive MPI ranks bound to a single physical GPU. Use 2 with Aurora compact tile binding. `balance=spread` distributes the finest level across GPUs and then their tiles, exchanging leaves with the smallest predicted increase in donor traffic. |
| `prepost_exchanges` | true | Compile peer demand from the actual stage graph and keep up to two receives posted per remote donor. False selects the probe-based comparison path. |
| `max_characteristic_cfl` | 0.35 | Upper bound on the local characteristic CFL at initialization, remesh and evolution stages. |
| `warmup_intervals`, `review_interval` | 32, 128 | Coarse endpoints for reviewing placement using measured kernel costs. |
| `partition_log`, `replay_file` | empty | Write or replay recorded endpoint placements. Use separate files when recording a replayed run. |
| `uniform_reference` | false | Advance all blocks at level zero with the same EX5 base method. Supply the fine reference timestep as `<time>/dt`. |

The complete four-region input is `small.athinput`. Output cadence expressed in
cycles counts coarse intervals. Set a uniform control's cycle-based refinement
and output periods to reach the same physical cuts as the multirate run.

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

## N12 comparison through 10 M0

The Aurora N12 comparison uses the 754-block atlas with two radial subdivisions in the penultimate region and four in the outermost. It reads the verified spatial input and binary initial-data manifest from the November 17 N12 campaign. Each run starts fresh in its own directory. The fixed base interval is (10/3)/1776 so both methods reach 10 M0 at a common endpoint. Native uniform classical RK4 takes 16 steps per base interval. The short numerical control also uses fine uniform EX5 with the multirate tableau.

Set the batch environment variable `MR_N12_MODE` to `uniform` or `multirate` in the benchmark's existing two-node Aurora debug job. The optional `MR_THROUGHPUT=1` first runs the placement and native uniform controls. Each invocation advances one segment, validates the accepted checkpoint, and records the executable and input digests. A subsequent invocation resumes from that mode's validated checkpoint. There is no submission loop.

The multirate trajectory uses the fastest measured placement among the tested candidates that reproduce the accepted payload. Complete local batches group blocks at the same numerical stage on one rank. The optional node-placement trial keeps each physical GPU's work and both tile packs together, measures rank-to-rank bytes, and enumerates placements of twelve GPUs across two six-GPU nodes. It compares the new placement with the original on hardware.

After both trajectories finish, `aurora/n12_compare.py` checks the endpoint fields, conformal regularity tensor, clock mismatch and constraints. It extracts Bondi news from the time series, compares the modes on common Bondi-time cuts, and reports elapsed time per M0 across all validated segments.


## Complete-batch transport controls

With `multirate/complete_local_batches = true`, `multirate/fused_rhs = true` waits for the batch's stage donors and evaluates its disjoint interior and halo-dependent bulk points in a single RHS pass. This option defaults to false. The SYCL bulk point kernel accepts `z4c_hyperboloidal/rhs_workgroup_size`. Its default is 128, and a value of 1 requests automatic sizing from Kokkos.

For separate interior and dependent passes, `multirate/wait_for_donors = true` holds the interior until the batch’s stage donors are ready. The `z4c_hyperboloidal/split_bulk_jets = true` option computes the ordinary bulk finite differences in a field-parallel pass and reuses scratch storage for the resulting jets. Batches that use the radial closure follow the original point kernel. Both options default to false.

On MPI builds with the SYCL backend, `multirate/progress_during_rhs = true` lets the solver thread service the existing record exchange while the GPU executes the bulk RHS. The validity reduction still completes before stage publication. This option defaults to false.

With `complete_local_batches = true` and `prepost_exchanges = true`, setting `coalesce_stage_packets = true` lets a rank send records from a common level, local step, stage and kernel class in one message per destination. The receiver posts two messages ahead per source and publication group. Each record keeps its own dependency key and consumer count. The shared allocation survives until its last record is released. Adaptation and redistribution rebuild the plan at common endpoints.

`send_cache_mib` and `receive_cache_mib` independently limit retained GPU buffer capacity per MPI rank in MiB. Both default to zero. Active communication storage is additional. The send cache recycles completed requests. The receive cache reuses an allocation only after the MPI request and every record view release it. Compare the `send_packets`, `reused_send_packets`, `reused_receive_packets` and peak GPU memory receipts at the same endpoint when tuning these controls.

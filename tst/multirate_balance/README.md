# Multirate partitioning tests

This standalone C++17 suite exercises the production balance implementation without Kokkos or an evolution run. It tests:

- measured batch-cost interpolation, launch overhead, noisy measurements and checkpoint roundtrips;
- parallel task dependencies, communication delay, batching and dependency-cycle rejection;
- improvement from vector phase costs when equal scalar work leaves ranks idle in alternate phases;
- sparse fast-region compaction when communication and batch occupancy favor fewer participating ranks;
- memory limits, deterministic METIS seeds, synchronized migration amortization and partition replay identity.

Configure with `cmake -S tst/multirate_balance -B <build> -DCMAKE_PREFIX_PATH=<METIS-prefix>`, build, and run `ctest --test-dir <build> --output-on-failure`. The library uses `ATHENA_MULTIRATE_METIS=1` when linked to METIS. A build without METIS tests the explicit error for requesting that backend and still exercises the scalar reference and scoring machinery.

## Adapter contract

`BlockCost::phase_costs` are predicted seconds per **base interval**, with distinct columns for relevant level/stage classes. Derive counts from the actual integration schedule; include projection, agreement and scri work. `work_units` denotes points or another consistently measured unit. GPU timing observations must cover completed work, not asynchronous launch overhead.

Pass the complete per-block stage dependency graph to `Partition` to enable dependency-aware batch scoring. Equal `ExecutionTask::phase` values assert that two tasks can legally share a batch; equal physical timestamps alone do not establish this. Dependency byte counts should represent actual payloads, accounting for any transport coalescing. Without a DAG, the score deliberately uses a conservative phase-synchronous model and reports no asynchronous overlap benefit.

The scoring model uses one compute lane per rank, overlapped transfers and a per-rank bandwidth lower bound. It is a partition-selection heuristic, not a runtime or network-congestion guarantee. Measure actual elapsed time and update profiles after warmup. Multi-constraint balance is softened for sparse phases, and candidate compaction is accepted only when its modeled completion time improves.

`ConsiderRebalance` is a policy decision only: the caller must reach a common accepted endpoint, drain outstanding communication, migrate accepted/persistent state, rebuild plans and record the selected partition. Replay records must override live profiling decisions when qualifying same-schedule bitwise restart. Graph fingerprints exclude measured timings but include GIDs, time levels, work/memory sizes and coupling topology; the containing checkpoint must additionally identify the integrator and physics configuration.

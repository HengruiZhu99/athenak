# N12 ownership evidence for later profiling

The actual qualified Aurora traces confirm whole-block ownership at 24 and 96
ranks. Their complete geometry keys agree with the bound N12 map and the current
native `Mesh::LoadBalance` assignment for 106 unit-cost blocks. This is a static
workload audit, not a timing result or a measurement of GPU utilization.

| Quantity | 2 nodes / 24 ranks | 8 nodes / 96 ranks |
| --- | ---: | ---: |
| Blocks per rank | 4–5 | 1–2 |
| Minimum evolved bulk points per rank | 6,912 | 1,728 |
| Mean evolved bulk points per rank | 9,558 | 2,389.5 |
| Maximum evolved bulk points per rank | 12,663 | 6,588 |
| Mean / maximum bulk point count | 0.7548 | 0.3627 |
| Ranks owning scri | 6 | 6 |
| Scri points on each surface owner | 144 | 144 |

Equal block weights do not imply equal evolved point counts. Depending on the
overlap and chart, a block owns 1,728, 2,160, 2,700, 3,375, or 4,860 bulk points.
At 96 ranks, rank 90 owns blocks 94–95 with 6,588 points. At 24 ranks, rank 8
owns blocks 32–35 with 12,663 points. The user-requested unit weights remain
unchanged.

If every bulk point took the same time and all other costs vanished, these
maximum counts would give a 1.9221 speedup and 48.05% efficiency for 24→96 ranks.
Those assumptions exclude communication, different core/shell arithmetic,
surface work, GPU occupancy and launch costs. The ratio is therefore neither
measured scaling nor an upper bound on the full implementation's scaling.

The surface work also does not shrink in this decomposition: each of the six
outer blocks stays intact and keeps its 144 points. Surface owners are ranks
11, 14, 16, 19, 21, 23 at 24 ranks, and 45, 57, 69, 81, 89, 95 at 96 ranks.
This is a reason to profile both a bulk-heavy rank and a surface owner after
the complete trajectory is admitted. It does not establish which rank sets
the elapsed time or whether surface arithmetic dominates.

The native source provides additional concrete profiling questions. Each of
the four bulk task batches returns a host validity reduction before completing;
the surface projection, normal solve and surface RHS also return host reductions.
The normal solve still evaluates the reference boundary RHS four times. Inspect
device execution, MPI progress and host waiting together before deciding which
operation to change. Summed kernel or phase maxima must not be reported as the
critical path. Keep detailed profiling separate from the validated whole-cycle
timing windows.

Two launch requirements remain for timing after the complete step-3551
trajectory is admitted. A restart retains the completed segment's terminal
`tlim`, so increasing only `nlim` will fail the timing preflight. The ordinary
restart command must set both the bounded cumulative `nlim` and a later `tlim`
covering the entire warmup and repeated windows, with at most 150 further steps.
Keep the timestep, physics, grid, equal weights and existing output cadence
unchanged; the harness must still reject any scheduled output inside that run.

The existing timing validator qualifies only the CPU two-rank, fresh eight-step
case. It does not admit actual 24/96-rank GPU timing. Future Aurora qualification
must bind the frozen numerical baseline, the same timing executable and restart
checkpoint hashes at both decompositions, the compiler/tile configuration,
actual state equivalence and complete window coverage. Neither the ownership
counts above nor the CPU validation timings establish GPU throughput or scaling.
The hash-qualified timing builder, header, contract and validator remain unchanged.

Reproducer: `tst/z4c_hyperboloidal/n12_ownership_audit.py`, using the actual
`native-clean` directories in the qualified two/eight-node stage-trace bundles
and `build-native/n12-checkpoint-map-aurora-v1`. The full result is
`build-native/n12-ownership-audit-v1/results.json`; its per-rank records and input
hashes are summarized in `evidence/native-n12-ownership-audit.json`.

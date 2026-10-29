# Local communication benchmark validation

The new standalone Kokkos/MPI benchmark passed local correctness and CLI
checks. **These use the Serial backend, with HostSpace for both named
memory routes; they establish no Aurora GPU performance result.** No
evolution or standard Z4c test was run.

| Local check | Coverage | Outcome |
| --- | --- | --- |
| One-rank CLI suite | Help, layouts, n16 all routes/modes/mix; 17 rejected invalid/overwrite calls | 20 commands passed their expected outcomes |
| Three-rank MPI | n16/n24/n32, unequal partitions, all routes, both barrier modes, 4-volume/14-surface mix | 270 measured rank-samples, zero mismatches |
| Twelve-rank MPI | n16 production ownership layout, six route orders, both modes and mixed calls | 936 measured rank-samples, zero mismatches |
| Two-rank CTest | Native n16 measurements and independent JSON/count/timing checks | Passed |

Every native warmup and measured workload validates all final receive
elements. The independent Python check also recomputed ownership layouts,
expected checked-value counts, each sample's maximum over ranks and the
reported quantiles. Staged validation checks both host and device buffers.
The optional mixed sequence validates final volume/surface buffers; its
intermediate gathers use identical payloads. All routes use separate
persistent allocations even though this backend places both in HostSpace.

[verification.json](verification.json) binds the source files, compiled
binary, commands and raw local report hashes. The original CLI and
three-rank check records and the two-rank CTest log are retained alongside
it. Full per-rank timing reports remain in the local output directory;
their CPU timings are not presented as GPU performance evidence.

The target was built with `-j64`. Reproduction commands and the precise
timing/validation contract are in the
[benchmark guide](../../mpi-communication-benchmark.md). Aurora still needs
a fresh bound build and a one-node/twelve-tile measurement before any
production communication change is considered.

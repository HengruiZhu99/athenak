# Native communication report contract: local validation

The benchmark now emits `status=completed`, matching the unchanged campaign
worker. It also emits `full_array_validation_passed`: **false for
layout-only output**, and true after all measured receive-array checks
pass on all ranks. The earlier native `status=passed` format was rejected
by that worker; this change fixes the integration without weakening the
worker's classification or checks.

Fresh Serial-backend checks passed after rebuilding with `-j64`:

| Check | Coverage | Outcome |
| --- | --- | --- |
| Single-rank CLI | 20 commands, including 17 expected invalid/overwrite failures | Passed |
| Three-rank MPI | n16/n24/n32, all routes/modes and mixed gathers | 270 measured rank-samples, zero mismatches |
| Twelve-rank MPI | n16, all six route orders, both modes and mixed gathers | 936 measured rank-samples, zero mismatches |
| Focused CTest | CLI and two-rank MPI, including actual worker classification | Both passed |

The focused test imports the unchanged worker classifier: measured reports
are accepted, layout-only output fails the explicit full-validation gate,
and the old native status remains rejected. Full-array validation, count/
displacement checks, per-rank timing checks and maximum-before-quantile
checks all remain enabled.

[verification.json](verification.json) records current source/worker hashes,
the fresh binary hash, commands, report hashes and validation counts.
Original local test records and CTest output are retained; complete timing
reports remain in the local output directory. Both named memory routes
use HostSpace on this Serial backend, so this evidence makes **no GPU
performance claim**. The [guide](../../mpi-communication-benchmark.md)
defines the Aurora measurement contract. The
[earlier local package](../communication-benchmark-local-v1/README.md)
is retained as evidence for its earlier native report format.

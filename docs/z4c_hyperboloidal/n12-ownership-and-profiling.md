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

The original timing validator qualifies only the CPU two-rank, fresh eight-step
case. The separate `native_cycle_timing_gpu_validate.py` reader now requires
the complete 3551-step admission and independent review, the frozen numerical
baseline and timing build, and matching restart checkpoint hashes at both
decompositions. It checks the compiler/tile provenance, complete state and
diagnostic equivalence, and complete window coverage before exposing accepted
timings. Neither the ownership counts above nor the CPU validation timings
establish GPU throughput or scaling. The hash-qualified timing builder,
header, contract and original CPU validator remain unchanged.

The same builder now compiled and verified a separate Aurora SYCL timing
executable while the frozen N12 trajectory continued. Its 11 reversible Driver
insertions preserve the original main and numerical objects; all 1,972 baseline
source files and 228 objects passed before/after checks. This removes a build
prerequisite only: no GPU timing run or numerical qualification was performed.
See [the actual build evidence](evidence/native-cycle-timing-aurora-build.json).

The first planned GPU timing qualification uses the same admitted cycle-3551
checkpoint for a normal and a timed run at each decomposition. The normal
executable remains `f3a85db7…`; the separate timing executable is `e4788800…`.
Each run advances 130 complete cycles with the original timestep: ten warmup
cycles followed by three 40-cycle windows. The only restart overrides are
`time/nlim=3681`, `time/tlim=2`, and `time/ndiag=1000000000`.

| Interval | Initial cycle | Final cycle | Complete cycles |
| --- | ---: | ---: | ---: |
| Warmup | 3551 | 3561 | 10 |
| Window 1 | 3561 | 3601 | 40 |
| Window 2 | 3601 | 3641 | 40 |
| Window 3 | 3641 | 3681 | 40 |

The current restart/output implementation and admitted cycle-1066 metadata
project the next diagnostic write after 3551 to cycle 3906. This prediction
must be checked against the actual admitted cycle-3551 checkpoint: both output
clocks should be `1.6666666666666667`, with restart file counter 11. The normal
restart entry skips initial outputs; finalization writes the ordinary history,
horizon and restart after the timing windows. Neither the output cadence nor
the saved horizon seed is overridden. The final time is
`1.7280161395080103`, below the proposed `tlim=2`.

Acceptance requires complete normal-versus-timed checkpoint equality at both
24 and 96 ranks, including all 31 fields, eight chi-record values, padding and
meaningful metadata; same-rank history/horizon/shape files must also match
byte for byte. Compare the timed final states across decompositions as well.
Bind the actual terminal jobs, commands, tile assignments, environments,
before/after runtime guards, and common source-checkpoint hash. Accept no
throughput or efficiency if any scientific or execution-provenance check fails.
The two allocations are serialized and retain the eight-node resource limit.
This is a concrete future run specification, not evidence of GPU execution.

The prepared `aurora/run-cycle-timing-validation.sh` and two PBS files execute
the normal/timed pair only after the runtime guard verifies the full admission.
The eight-node entry also checks that the two-node job is terminal F/0 and used
the same guarded inputs. Submission remains separate and requires a fresh queue
check. The guard's module docstring specifies the hash-bound `admission.json`
descriptor; no final-3551 descriptor has been fabricated or staged.
`aurora/collect_cycle_timing.py` collects complete final outputs and compact
provenance without duplicating the source checkpoint or numerical build.

Offline controls exercise actual existing CPU payloads and outputs, the actual
SYCL build proof, and explicitly synthetic admission/window/environment/metadata
cases. The public reader rejects the real incomplete cycle-1066 admission and
withholds all timing values. These checks do not exercise a successful complete
four-run GPU collection; that remains the next qualification after 3551.
See [preparation evidence](evidence/native-cycle-timing-gpu-preparation.json).

An installed-tool audit found unitrace 2.3.0 with MPI, Level Zero and SYCL trace
options, plus VTune 2025.8.0 and APS under the pinned compiler module. This is
availability evidence only; no native GPU collection has run. Loading the
`pti-gpu/0.17.0` module changes the solver's resolved `libstdc++` and `libgcc_s`
to byte-different libraries. Prefer an absolute profiler path under the existing
baseline environment, then verify actual loaded libraries during a separately
qualified collection. A successful `--version` command does not prove that
the combined GPU/MPI collector works.

After the unprofiled results, a conditional first trial is four complete cycles
from 3551 on two selected ranks per decomposition: bulk-heavy rank 8 and scri
owner 23 at 24 ranks, or ranks 90 and 95 at 96 ranks. These are static workload
candidates, not established critical ranks. Use the frozen normal executable,
preserve the timestep/output cadence, and compare complete final state and
ordinary diagnostics against matching unprofiled four-cycle controls. The
130-cycle timing controls end at a different state and cannot substitute.
No such profile or control is staged or submitted; the actual timing results
will determine whether this trial is appropriate.

The initial collection should avoid hardware metrics and all-rank Chrome MPI
logging. `MPI_Testall` polling can generate large traces and perturb progress;
the event-buffer size does not cap total output. Kernel names may identify
compiler functors rather than Kokkos runtime labels, and all four bulk RHS
subsets share a callsite. Selected-rank traces can expose local GPU/host waits,
but neither overlapping nonblocking calls nor merged timestamps establish a
whole-job critical path. The installed merge helper does not align clocks and
can modify malformed inputs, so preserve raw files. Plan remote retention and
bounded transfer before collection: four additional checkpoints alone total
about 791 MB, beyond the space already needed for the remaining trajectory and
timing data. See [tool/readiness evidence](evidence/native-profiler-readiness.json)
for exact installed paths, hashes, commands and interpretation limits.

Reproducer: `tst/z4c_hyperboloidal/n12_ownership_audit.py`, using the actual
`native-clean` directories in the qualified two/eight-node stage-trace bundles
and `build-native/n12-checkpoint-map-aurora-v1`. The full result is
`build-native/n12-ownership-audit-v1/results.json`; its per-rank records and input
hashes are summarized in `evidence/native-n12-ownership-audit.json`.

# Aurora Phase-8 fenced-cycle PVC fault

Aurora debug job `8791465` ran the source-identical Phase-7 executable on
node `x4117c6s7b0n0`, using 12 ranks on 12 distinct PVC tiles.  The only
runtime diagnostic change was `ref_gh/debug_task_fences=true`; the run was
limited to one RK cycle and all restart/field outputs were disabled.

The task-fence counts show 60 completed instances of every CalcRHS subkernel,
60 completed metric and gauge physical-boundary kernels, 48 RK updates, and
12 completed final timestep reductions.  The Level Zero `NotPresent` write
fault then recurred on multiple ranks.  PBS recorded exit 143 after 74 seconds
because `mpiexec` terminated the remaining ranks after device-runtime aborts.
No positive-time history sample was written.

The interleaved multi-rank stdout cannot establish which earlier kernel, MPI
operation, or view lifetime originally corrupted the address.  In particular,
the last printed line is not a reliable per-rank failing boundary.  The result
does show that the fault is not raised at the analytic-reference, CalcRHS, RK,
boundary, or timestep Kokkos fences.  The next useful check is bounds/ASan
instrumentation, not another unfenced 3M launch or parameter tuning.

`remote_compact_sha256.txt` is the original in-job manifest and includes the
1.63-MB duplicate `fence_trace.txt` retained at the remote path.  That duplicate
is intentionally omitted from Git because the complete trace is already
present in `run_cycle/run.log`; the repository `SHA256SUMS` validates the
committed bundle.

Complete Aurora directory:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_fully_subtracted_gauge_repair_20260829_3c9a34c8_phase7_v1/runs/phase8_cycle_localize_8791465.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

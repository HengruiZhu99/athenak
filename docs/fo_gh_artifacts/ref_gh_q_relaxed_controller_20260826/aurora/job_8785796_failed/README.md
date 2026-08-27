# Aurora job 8785796: failed dynamic-cache discriminator

This is compact evidence from the single bounded Aurora gate run from commit
`f184fcde3d53b0bed36a855c2a2e07c515abf594`.

- PBS queue: `debug`
- project: `CompactBinaryMerger`
- node: `x4201c3s2b0n0`
- mapping gate: 12 MPI ranks on 12 distinct PVC tiles
- evolved layout: eight MPI ranks on eight PVC tiles
- wall time: `00:09:29`
- exit status: `134`
- retained campaign root:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260827_f184fcde_v2`

The source-unit/cache gate passed. The full-output evolved gate completed its
first RK stage on all eight ranks, including the staged q-estimator kernels,
RHS, RK update, communication, prolongation, and physical boundaries. All
ranks then completed the next `CopyU` and q-estimator sample/sum/extrema tasks,
but none reached the next `UpdateReference` fence. They failed with a Level
Zero `NotPresent`, PDP-level GPU write fault.

This discriminator rules out the former mixed combined q reduction as the
established root cause. The failure is localized to the first dynamic
`FillReferenceCache` rebuild at the next RK stage. Static q source-unit tests
set `reference_time_dependent=false`, so they do not cover this exact path.
The specific failing cache subkernel is not yet known.

`cycle_run.log` is the complete focused runtime trace. `provenance.txt`,
`configure.log`, `rank_gpu_mapping.txt`, `source_unit.log`, and `qstat.txt`
preserve the source, build, device, scheduler, and termination evidence. No
restart or field dump is committed.

This is failure-localization evidence only. It does not establish evolved PVC
qualification, moving-reference convergence, closed-loop relaxation, or
production readiness.

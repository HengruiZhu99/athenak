# Aurora job 8785833: provider-profile kernel localization

This is compact evidence from the single bounded Aurora discriminator at
commit `bd40d98b64c4a124e5b8c36679c71a62d1dc6071`.

- PBS queue: `debug`
- project: `CompactBinaryMerger`
- node: `x4118c0s1b0n0`
- mapping gate: 12 MPI ranks on 12 distinct PVC tiles
- evolved layout: eight MPI ranks on eight PVC tiles
- wall time: `00:10:13`
- exit status: `134`
- retained campaign root:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260827_bd40d98b_v1`

The job was submitted with `CYCLE_GATE_ONLY=1`; the production pulse ladder
could not run. The source-unit/cache gate passed. During evolved startup, all
eight ranks passed every newly labeled reference-cache subkernel. The first RK
stage then completed. At the next stage, all ranks completed the q estimator,
but no rank reached the next `ref_gh reference provider profiles` fence. Every
rank failed with a Level Zero `NotPresent`, PDP-level GPU write fault.

This localizes the first failing kernel to the q-controlled provider-profile
launch in the first dynamic cache rebuild. It rules out the downstream frame,
connection, mixed-gauge, theta, spin, and curvature kernels as the first fault
in this run. It does not by itself prove whether the write is to the provider
view or device-private spill/scratch memory. The q-controlled provider kernel
currently constructs three 33-Real jets per work item, so excessive private
state is a portability hypothesis only, not an established cause.

`cycle_run.log` is the complete runtime trace. The remaining files preserve
source, build, rank-to-tile, scheduler, and source-unit evidence. The build
excerpt is compact; the full build log remains at the campaign root. No field
dump or restart is committed.

No evolved PVC qualification or scientific convergence claim follows from
this failed discriminator.

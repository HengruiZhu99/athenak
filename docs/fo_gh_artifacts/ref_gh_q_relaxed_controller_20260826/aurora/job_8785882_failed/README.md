# Aurora job 8785882: specialized q-provider gate failure

This is compact evidence from the single bounded Aurora discriminator at
commit `5ac381803ed1ac04da24ff626e4cda1ac57564f8`.

- PBS queue: `debug`
- project: `CompactBinaryMerger`
- node: `x4015c7s5b0n0`
- mapping gate: 12 MPI ranks on 12 distinct PVC tiles
- evolved layout: eight MPI ranks on eight PVC tiles
- wall time: `00:10:02`
- exit status: `134`
- retained campaign root:
  `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260827_5ac38180_v1`

The job was submitted with `CYCLE_GATE_ONLY=1`; no production pulse could run.
The source-unit/cache gate passed. Initial cache construction passed every
labeled subkernel on all evolved ranks, and the first RK stage completed. At
the next stage, all ranks passed the staged q estimator but none reached the
next `ref_gh q-controlled provider profiles` fence. All ranks failed with a
Level Zero `NotPresent`, PDP-level GPU write fault.

Commit `5ac38180` had specialized this provider launch so each work item
evaluated and stored only one of the three profile jets and did not capture the
unrelated controlled/generic provider state. Local source-unit and full-cycle
tests passed, with principal nonzero-history differences limited to roundoff.
The identical PVC failure means that refactor was insufficient and disproves
the original three-simultaneous-jet capture as a complete explanation. The
first failing launch remains the q-controlled provider-profile kernel; the
specific failing write is unresolved.

`cycle_run.log` is the complete runtime trace. The remaining files preserve
source, build, device mapping, scheduler, and source-unit evidence. No field
dump or restart is committed.

This failed gate does not establish evolved PVC qualification, moving-reference
convergence, closed-loop relaxation, or production readiness.

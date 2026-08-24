# Aurora T5 result, job 8777824

This directory contains compact evidence from the required aggressive
prescribed-tau-4 discriminator on the same `[-24M,24M]^3` medium SMR mesh as T4.
The job used eight Aurora nodes, 96 distinct PVC tiles, and the audited
328-MeshBlock tree.  Checkpoint/restart continuity passed.

The segment targeting t=3.5M failed at stage time 3.18677M because the Ref-GH
relative metric/conditioning became invalid.  The last history output is
t=3.150078211271024M and xi=0.7875195528177561.  T5 did not reach xi=1 or
t=4M.  This is scientific failure evidence; later Kokkos finalization messages
are consequences of the deliberate fatal exit.

`t5_summary.json`, `t5_acceptance.json`, and `t5_first_bad_state.json` contain
the compact numerical result.  Logs, histories, max locations, mapping,
provenance, mesh evidence, and hashes are preserved here.  Large field and
restart files remain at:

```
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_feedback_continuation_20260823_2466879e_v1/runs/t5_prescribed_outer24_8777824.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

# Aurora T4 result, job 8777607

This directory contains compact evidence from the definitive medium-resolution
closed-loop run on the `[-24M,24M]^3` SMR mesh.  The job used eight Aurora nodes,
96 distinct PVC tiles, and the audited 328-MeshBlock tree.  Segmented restart
continuity passed, but the frozen safety policy stopped the evolution at
t=3.70126M before xi reached one.  This is a T4 failure and prohibits T6.

The event sequence and exact state values are in `t4_first_bad_state.json`;
overall numerical and acceptance summaries are in `t4_summary.json` and
`t4_acceptance.json`.  Raw compact histories, max-location tables, logs,
rank-to-tile mapping, build/configuration records, and hashes are retained.

Large restart and field files were deliberately not copied into Git.  They
remain at:

```
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_feedback_continuation_20260823_2466879e_v1/runs/t4_feedback_outer24_8777607.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

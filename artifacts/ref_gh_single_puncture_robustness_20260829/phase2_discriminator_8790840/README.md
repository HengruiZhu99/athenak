# Phase 2 12-tile discriminator: Aurora job 8790840

This run retained the identical 96^3 physical grid and every equation/numerical
parameter from failed job 8790836, but changed the decomposition from 216 ranks
with one MeshBlock per rank to 12 ranks with 18 MeshBlocks per rank.

It reproduced the same fail-closed invalid effective timestep at cycle 330,
`t=1.123484M`.  The last accepted history time is `t=1.000922M`; PBS exit
status is 143.  All eight history streams agree with job 8790836 to a global
conditioned Linf of `1.5729630671258575e-13`, and the user/controller history
is identical.  See the sibling `phase2_decomposition_comparison.json`.

This makes the observed failure independent of the two tested MPI
decompositions.  It does not distinguish a formulation defect from an
equation-implementation defect, and it provides no stability evidence.

The initial binary64 health record passes, but the run failed before the 2M
accepted-state output cadence and therefore has no positive-time offline
signature record.  Live histories show exponential native-constraint and
source growth before the fatal guard.

Large restart and binary field payloads remain uncommitted at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_single_puncture_robustness_20260829_253c78e2/runs/phase2_matched_q100_h24_r12_8790840.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

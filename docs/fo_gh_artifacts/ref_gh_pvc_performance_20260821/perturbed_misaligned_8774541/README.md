# Rejected nominal-time comparison (Aurora job 8774541)

This first binary64 run passed numerically, but its nominal t=0.2 and t=0.4
outputs occurred after the requested intervals and at different physical times
for 64³, 96³, and 128³.  For example, the first triplet was at 0.204722,
0.200865, and 0.201495.  Those two comparisons are therefore rejected as
convergence evidence.  The common t=0.6 endpoint was valid and fourth order,
but the exact-time rerun in `perturbed_exact_binary64_8774567/` is authoritative.

The full run is retained at
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/perturbed_binary64_8774541.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.

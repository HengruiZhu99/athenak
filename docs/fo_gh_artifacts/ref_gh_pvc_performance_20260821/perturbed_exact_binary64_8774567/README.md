# Exact-time binary64 convergence (Aurora job 8774567)

This is the authoritative 64³/96³/128³ perturbed-trumpet convergence ladder.
Independent runs terminate exactly at t=0.2, 0.4, and 0.6.  Both the ten-field
perturbation and six native constraints use binary64 cbin payloads and
sixth-order interpolation to a fixed r<1 sample grid.

Aggregate field L2/Linf orders are 4.4019/4.9647, 4.1973/4.7000, and
4.0610/4.5094.  Native-constraint L2/Linf orders are 5.4195/4.1201,
4.4279/4.1427, and 5.4752/4.1533.  Psi L2 orders are 4.7628, 3.6955, and
3.5714; its pointwise Linf order is lower and is reported separately rather
than hidden.  All eight ranks used distinct PVC tiles.  The maximum measured
characteristic speeds imply an earliest outer-boundary arrival to r<1 of
t=1.632--1.638, later than every analyzed endpoint.

The full cbin fields and restarts are excluded from Git and remain at
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/perturbed_binary64_8774567.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.

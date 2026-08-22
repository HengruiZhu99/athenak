# Final hardened-source PVC gate (Aurora job 8774480)

This compact directory is the current runtime authority for source commit
`afc5b6b61cc3b56924cd198b80a61b57c8a8c82c`.  It includes the lower-dimensional
extent and fail-closed ADM reconstruction hardening after the source/Pi fusion.

The one-tile bounds-check-off benchmark reached `1.272989e6` active
zone-cycles/s.  The 3.93% difference from job 8774435 is treated as independent
job variability because the hardening does not alter the timed three-dimensional
hot path.  The independent source oracle, initial RHS, full-output checked gate,
time-dependent five-stage cache test, restart, and performance-output gate all
passed.  Checked and performance histories differed by at most
`1.0913293364176594e-16`.

The synchronized profile assigns 59.86% of completed kernel time to the fused
source/Pi kernel.  The checked/performance executable SHA-256 values are
`7cf89edd636ce753f0f9dd2d9485288049c52b5be80c1273072dcded3a81854f` and
`3d1fa89001374748a93ddcfa44513e4c0531af9373f353c2d5161d361e0cd70c`.
Large restarts, binaries, build logs, and profiler libraries are intentionally
excluded; the full run remains at

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/robust_final_8774480.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

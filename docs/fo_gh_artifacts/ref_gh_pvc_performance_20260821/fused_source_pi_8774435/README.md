# Fused source/Pi PVC gate (Aurora job 8774435)

This compact directory records the first checked and performance gate for the
equation-neutral Ref-GH scalar-source/Pi fusion.  The source state was committed
immediately afterward as `17ab455f`; the recorded checkout authority is
`5fd1a72f` plus that intended `ref_gh_calcrhs.cpp` change.

The one-tile bounds-check-off benchmark reached `1.325066e6` active
zone-cycles/s, `1.3813909737x` the preceding lean-source result.  The independent
source oracle reported flat/nonflat maxima `5.55112e-17` and `3.33067e-16`.
The checked full-output cycle reported initial-RHS Linf `1.10048e-16`, field
Linf `9.992007e-16`, and native-constraint Linf `2.461044e-14`.  Checked and
performance histories differed by at most `1.0913293364176594e-16`.

The synchronized profile assigns 59.83% of completed kernel time to the fused
source/Pi kernel.  `compiler_pressure.tsv` preserves the compiler retry and
explicit checked-build spill records.  Large restarts, binaries, build logs,
and profiler libraries are intentionally excluded; the full run remains at

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/fused_source_pi_8774435.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

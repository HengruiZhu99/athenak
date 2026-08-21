# Ref-GH lean production source: Aurora job 8774182

This compact artifact set records the final passing PVC performance gate for
source commit `fc2d61adf3559b5c09707a50cb6ddbc2b612bc26`.  The PBS job completed
with exit status zero on node `x4220c4s7b0n0`, tile `0.0`, using the Level Zero
PVC device.  No campaign job remained queued or running at wrap-up.

The production-only scalar-source path computes the same ten evolved source
components as the retained full diagnostic/oracle path.  The independent PVC
oracle reported maximum discrepancies of `5.55112e-17` over 1000 flat states
and `3.33067e-16` over 128 nonflat states.  The checked stationary test reported
cache-oracle Linf `2.36929e-14`, initial-RHS Linf `1.10048e-16`, one-cycle field
Linf `9.992007e-16`, and native-constraint Linf `2.461044e-14`.  The
time-dependent provider rebuilt at five stage times and ended with maximum
field error `9.992007e-16`.

The output-free `64^3`, one-MeshBlock, one-tile benchmark achieved
`9.592259e5` active zone-cycles/s.  This is `2.70189865788x` the preceding
`3.550192e5` diagnostic-split measurement and `8.3391x` the original
`1.150275e5` Ref-GH baseline.  The mature matched Z4c control remains
`7.4389x` faster.  The synchronized final profile assigns 35.24 percent to the
scalar source and 32.42 percent to Pi RHS.

The full remote run remains at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/lean_source_8774182.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

Large restart files, binaries, profiler libraries, and complete build logs are
not committed.  `provenance.txt`, configure logs, GPU mapping, executable
hashes, source/check logs, histories, throughput, and kernel timing tables are
included for audit.  This is a correctness/performance gate, not a convergence
or long-time stability result.

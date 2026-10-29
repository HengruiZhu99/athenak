# GPU boundary-reuse preflight and timing

Job 8986786, terminal F/exit 0, one debug node, CompactBinaryMerger allocation,
12 MPI ranks on 12 unique GPU tiles. Frozen source c97706ab7, source-v20;
SYCL+MPI Release build with make -j64. Independent verification passed all
340 checks and 48 artifact identities, including whole-file checkpoint equality
for pulse restart and the matched 100-step puncture execution pair.

Native times: control 74.386 s, fast 71.445 s. This is a modest 4.0% wall-time
reduction in one same-node pair. Bulk RHS still accounts for only 16.8%; global
volume gathers remain the main measured communication cost. See the analysis
in ../../boundary-reuse-v1 for exclusive phase costs and limits.

This preflight does not certify long-time stability, convergence, or a settled
trumpet. Raw checkpoint/archive copies are retained locally; redundant remote
large copies are purged after hash verification to keep home usage below 5 GB.

After verified cleanup, remote campaign footprint: 4271169536 bytes.

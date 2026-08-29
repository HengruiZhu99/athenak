# Aurora Phase-8 cycle-zero PVC write fault

Aurora debug job `8791456` ran commit
`54881e4e17c10f50f29e71691a6df4c7228d5feb` on node
`x4507c1s2b0n0`, project `CompactBinaryMerger`.  It reused the Phase-7
production executable built from source-identical commit
`3c9a34c8c3123c2570eb33e8ec77368feb1f1c61`, SHA-256
`9a9eae63fc11fd0448570a2bd04dfd7c08b4100aa4b75c65122beb1d10744537`.
All twelve ranks mapped to distinct PVC tiles `0.0` through `5.1`.

The exact 96^3 STANDARD matched state initialized successfully.  The
cycle-zero residual target remained exact zero, the stored Hhat/theta fields
were exact zero, and the initial total RHS was `3.1002e-13`, with the gauge
RHS families zero.  Immediately after reporting cycle 0, before any positive
time was reached, every rank encountered a Level Zero `NotPresent` GPU write
page fault and the Intel runtime aborted.  PBS recorded exit 134 after 54
seconds.  Therefore this is a PVC task/view portability failure, not evidence
for or against recurrence of the old GH exponential mode.

The launcher's inherited ERR trap stopped the postprocessor before it could
write its normal status and manifest.  The derived status file and the
repository `SHA256SUMS` preserve this transparently; the launcher must be
corrected so an expected nonzero simulation exit still reaches evidence
collection.  No 5M continuation was launched.

The uncommitted 1.46-GB cycle-zero restart remains on Aurora at the location
recorded in `remote_large_files.txt`.  It is not a valid positive-time
continuation checkpoint and is not committed.

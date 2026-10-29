# V11 GPU preflight v2 — passed

Job 8986460 repeats the same seven cases on one debug node with 12 ranks/tiles,
CompactBinaryMerger. It corrects exactly two inherited V10 metadata expectations
in the previous plan's restart audit (magic and header size). All numerical
criteria and the frozen source-v19 binaries are unchanged. 257 gates are checked.

Source: faee66ea34d8d9b7a1b0412abab09caf515c03b5. The SYCL/MPI build passed in
260 seconds; an independent audit matched 2645 source/Kokkos hashes to Git.
The independent checkpoint reader checks all four state/compensation arrays,
V11 geometry and positive-variable descriptors, exact restart, and artifact hashes.
Submission alone is not a preflight or stability admission.

Independent verification-v3.json passed all 257 gates and 39 artifacts. Initial
reader attempts exposed a stale default repository path and an inherited gate
count; both were corrected to the already pinned plan, without changing any
scientific criterion. Earlier reader outputs are retained. The GPU short puncture
has zero lapse-floor interventions; null closure remains at roundoff.
All checkpoints are archived locally; remote duplicate payloads were purged after
hash verification. This admission is for bounded implementation/restart checks,
not duration stability.

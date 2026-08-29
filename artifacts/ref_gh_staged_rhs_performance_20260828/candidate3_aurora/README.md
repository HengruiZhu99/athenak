# Candidate 3 Aurora coordinate-source discriminator

Candidate 3 is rejected.  It passed every deterministic local oracle and both
local evolved comparisons at unchanged tolerances, but its required Aurora PVC
full-output gate failed reproducibly before the first evolved physical-boundary
fence completed.

- Exact source tested: `7a8d67a73a5a81ae7998dd666245bb0c588ac36a`.
- Job 8790590 on `x4504c1s3b0n0`: exit 134, PVC write page fault.
- Source-identical retry 8790608 on `x4514c1s0b0n0`: exit 134, same phase and
  same PVC `NotPresent` write fault.
- Both jobs used queue `debug`, account `CompactBinaryMerger`, one node, and
  mapped eight ranks to eight distinct PVC tiles before the one-rank gate.
- In both jobs the first evolved CalcRHS, RK update, restriction,
  communication, and prolongation fences passed.  The process faulted before
  `ApplyPhysicalBCs projected trumpet metric` reported completion.
- The eight-rank evolved run was never reached.  No numerical comparison,
  benchmark, or runtime profile was run after either failed gate.

A fresh local Debug/Kokkos-bounds-check build completed the full four-stage
one-cycle workload, including every physical-boundary and diagnostic fence.
That rules out a bounds error visible to Kokkos's host view checker, but does
not qualify the PVC candidate or explain the device write fault.

The compiler report is diagnostic evidence only: the coordinate frame-
transform flat kernel reported 10 spilled Reals for FD4 and 28 for FD3.  The
large four-index-kernel warnings remain ambiguous without compiler-emitted
kernel labels, so they must not be used to attribute runtime cost to a named
kernel.  A profiler was intentionally not run after the failed PVC gate.

Full remote evidence remains under:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260829_candidate3_7a8d67a7`


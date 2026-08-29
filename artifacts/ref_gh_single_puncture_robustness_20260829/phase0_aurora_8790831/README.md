# Phase 0 Aurora PVC gate: job 8790831

Aurora debug job `8790831` ran on node `x4705c2s0b0n0` from source commit
`253c78e2ccb521860deaeb77f810db35f4f7c924`.  It used the source-frozen
analytic radial-q production backend, Kokkos SYCL, GPU-aware MPI, and eight
distinct Intel PVC tiles (`0.0` through `3.1`).

Both the one-rank/one-tile and eight-rank/eight-tile dynamic-q evolved cycles
passed.  Their conditioned Ref-GH history Linf difference was
`3.88980825583101983e-14`, below the pre-existing `5e-12` tolerance.  The
production image reported zero generic reference-cache bytes and exactly the
12-Real static / 8-Real stage analytic allocation.

The PBS exit status was zero.  This is the required bounded Phase 0 PVC gate;
it is not long-time robustness, convergence, or performance evidence.

The full large build tree remains on Aurora at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_single_puncture_robustness_20260829_253c78e2`

The executable SHA-256 was
`b247762c83e5bba2b8a5331f9ce372d17462eb6f7ef59c63b4ddef39306a05e0`.

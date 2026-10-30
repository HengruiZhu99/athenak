# Field-batched radial derivatives — startup storage guard rejection

Frozen source: `146fb4b251274c4a5f90ee57b688ae63e73e510a` (source-v24).
Manifest SHA256: `15205ac9cec832ee6e72b04759b1209fc58e81ddcc58cd6dd7470d3eecb77b67`.

The candidate moves the existing radial derivative sums into a bulk kernel over
selected radial rows and fields. Stencil indices and weights remain persistent;
field derivatives refresh on each call for the current stage. Equations, donor
summation order, log chi/lapse policy, physical lapse floor, nonlinear null
closure and V11 checkpoint schema are unchanged. The source target remains
explicitly experimental; its local equivalence results do not establish GPU
speed or GPU numerical equivalence.

The retained plan contains eleven cases and 419 checks: sparse-plan lifecycle,
full native operator/device parity, direct versus batched radial derivative
parity, Minkowski evolution, pulse and restart, and a same-node 100-step n16
puncture pair. The control is the independently verified frozen V23 profiled
sparse executable. All twelve MPI ranks/GPU tiles run on one debug node under
CompactBinaryMerger. Build uses SYCL/MPI and `make -j64`.

Both pulse restart and control/candidate puncture checkpoints must match in
full. The new radial component also requires exact direct-device equality for
110,019 field-row comparisons, three stage refreshes per plan, noncontiguous
and empty ownership, exterior-ghost poison and shape/index rejection coverage.
The independent reader checks all source blobs, executable/cache/launcher pins,
raw artifacts and timer accounting. Its 21 offline negative-control tests pass.

The source export and build scripts are retained in `build-v24/`. Preparation
verified 2,654 source blobs and a campaign footprint of 4,321,085,941 bytes.
The build passed in 296.30 seconds. Independent verification checked all 2,654
Git source blobs, all four named executable hashes and the SYCL/MPI/PVC build
configuration. The post-build footprint is 4,356,928,588 bytes. Job **8987469** was submitted after independent binding verification.
No GPU test result or performance gain is claimed yet. The
storage guard enforces the 5 GB cap. Long-time evolution, horizon-quality,
convergence and coherent all-operator mesh-change lifecycle remain separate
outstanding requirements; this campaign does not establish AMR evolution.

## Terminal outcome

Job 8987469 finished in PBS state F, exit 1, after 1:46. The worker
rejected the prospective storage budget before creating its status file or
running affinity, correctness or evolution tests. The root footprint at worker
startup was 4,404,302,336 bytes; adding the unchanged 520 MB output allowance
and 100 MB atomic-checkpoint reserve exceeded 5 GB by 24,302,336 bytes.
The footprint had grown after submission. No test failure or speed result can
be inferred. The native stdout and scheduler/release records are retained.

A fresh [v2 retry](../preflight-radial-field-batch-v2/README.md) keeps the same
frozen source, binaries, physics, input, checks and resource limits. Verified
redundant old stability checkpoints/failure captures were removed to give
it substantially more storage headroom; the retry also uses a larger allocated-byte
output estimate. The failed plan is not reused.

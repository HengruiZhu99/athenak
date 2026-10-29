# Sparse stage exchange preflight

Prepared against frozen source `fd522df9248c4911c9352057156b3ec58e7dc02b`.
Source-v21 manifest SHA256:
`5870a4f1290ded6e9cca2a6a592276119e1b46e5c52ee61fad71c8ba07e94fd2`.
The SYCL/MPI build is in progress. This directory does not yet contain a GPU
result or admission. Do not treat the reviewed test plan as a passing run.

The plan uses CompactBinaryMerger, one debug node, 12 ranks and GPU tiles.
It checks the donor-plan lifecycle, native CPU/device component parity,
Minkowski, pulse restart, and a 100-step puncture pair against the frozen V20
uncached V11 executable. All 409 gates must pass, with exact whole-checkpoint
comparisons. No scientific acceptance threshold has been relaxed.

The independent verifier binds candidate build identities and the separately
verified frozen control. Binary pins will be recorded after the build and before
submission. Its 20 offline tests check checkpoint layouts, array coverage,
exact comparison, configuration, placement, pins, and frozen control identity.

The preparation cleanup removed only eight obsolete checkpoints after checking
both their local copies and remote SHA256 values. The receipt retains their
identities and local locations. The current 3M and parent checkpoints remain
available on Aurora. Build and output budgets reserve atomic checkpoint space.

The build scripts are the executed frozen-source preparation workflow. The
export script is intended to run from the local `output/hyperbolic-core-shell/`
`aurora-staging/` directory; the remaining scripts run in the recorded Aurora
home campaign root. Compilation uses `make -j64` and oneAPI 2025.3.1.

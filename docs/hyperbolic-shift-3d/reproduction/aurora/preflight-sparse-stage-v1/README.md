# Sparse stage exchange preflight

Prepared against frozen source `fd522df9248c4911c9352057156b3ec58e7dc02b`.
Source-v21 manifest SHA256:
`5870a4f1290ded6e9cca2a6a592276119e1b46e5c52ee61fad71c8ba07e94fd2`.
The SYCL/MPI build passed in 428.60 seconds. Independent verification matches
all 2,649 manifest entries to frozen main/Kokkos Git blobs and pins all three
executables. Job **8987177** completed all ten cases. Independent verification replayed all
409 checks and verified 49 artifacts, including exact complete-checkpoint
restart and execution comparisons. This is a preflight, not a long-time
scientific admission.

The plan uses CompactBinaryMerger, one debug node, 12 ranks and GPU tiles.
It checks the donor-plan lifecycle, native CPU/device component parity,
Minkowski, pulse restart, and a 100-step puncture pair against the frozen V20
uncached V11 executable. All 409 gates must pass, with exact whole-checkpoint
comparisons. No scientific acceptance threshold has been relaxed.

The independent verifier binds candidate build identities and the separately
verified frozen control. Binary pins were recorded and independently checked after the build and before
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

## Measured result

Same-node n16 puncture pair, 100 steps, 12 GPU tiles, identical input/dt/diagnostics:

| Quantity | Frozen uncached V11 | Sparse + boundary reuse |
|---|---:|---:|
| Native elapsed time | 74.504 s | 45.435 s |
| Seconds/step after initial diagnostic | 0.63006 | 0.32860 |
| Bulk RHS, longest accounted rank | 11.953 s | 12.026 s |
| Recorded communication/waiting, same rank | 27.560 s | 1.441 s |

This pair gives 1.64x native wall speedup and 1.92x after-initial-diagnostic step
speedup. The latter includes the final diagnostic. The experiment combines
sparse exchange with the already tested boundary reuse; it does not isolate
sparse exchange against a fresh same-node fast-only control. Logical remote
values decrease by 53.66%. These are request counts, not measured link traffic.

The native physics and RK compensation arrays match bit for bit. All floor
counts remain zero; null errors meet the original precision gate. Bulk RHS
cost itself is unchanged and still differs by 2.31x between ranks. Boundary
work/checking and startup now account for much of the remaining time. Recorded
MPI timers do not include every standalone synchronization, so residual time
must not be described as pure setup. Measured ownership and finer boundary
profiling remain follow-up work. No convergence, long-time trumpet, or AMR
claim follows from this preflight.

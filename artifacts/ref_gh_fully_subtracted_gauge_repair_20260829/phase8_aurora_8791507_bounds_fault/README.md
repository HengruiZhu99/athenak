# Phase-8 Aurora PVC bounds-gate failure: job 8791507

Aurora debug job `8791507` ran the focused one-cycle bounds/lifecycle gate at
source commit `c3ead1836e5c69aceab277e092a8dd2b81149c13` on node
`x4220c4s5b0n0`. PBS charged `CompactBinaryMerger`, used one node for
`00:19:26`, and recorded exit status 143. The executable SHA-256 is
`a1fdeaded8576135123c3d3547c6ef2498791cbd8aea5b8bd7eac1e92b75cb25`.

The build used IntelLLVM 2025.3.2, Kokkos 4.7.2, MPI, SYCL, PVC architecture,
and `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON`. All twelve ranks reported distinct
Level Zero PVC affinities `0.0` through `5.1`. Cycle-zero initialization,
constraints, history reductions, physical boundaries, and communication
cleanup fences completed. The only history row is at `t=0` and is finite.

The first evolved stage did not reach `ExpRKUpdate`. Ranks 1--11 printed the
post-`Kokkos::deep_copy(u_rhs, 0.0)` `CalcRHS zero` fence; rank 0 did not.
Level Zero then reported multiple level-2 `NotPresent` GPU write page faults.
Because rank stdout is interleaved and other ranks may already have entered the
next kernel, this identifies a stage boundary but does not prove which earlier
kernel caused latent corruption. Kokkos emitted no bounds diagnostic.

The build log reports approximately 1289--1296 spilled Reals for the analytic
active-cell RHS kernel in the FD2/FD4/FD6 instantiations and 355--373 for the
analytic Psi kernel. This is direct compiler evidence of extreme PVC private
working-set pressure, but is not by itself proof of the page-fault cause. The
full 1.5 MiB build log is retained only at the Aurora location below; its
SHA-256 is recorded by `compact_sha256.txt` as
`d84eff9857a00cffde4a18440c99afac357ebbe45c6366f4a74071d3a9df4c67`.

Aurora source/build/output location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_phase8_bounds_20260829_c3ead183_v1`

This is a failed portability gate. It supplies no positive-time evolution,
stability, or convergence evidence. No follow-on production run was launched.

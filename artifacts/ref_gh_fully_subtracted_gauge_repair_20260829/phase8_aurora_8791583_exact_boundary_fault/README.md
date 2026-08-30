# Phase-8 Aurora PVC exact-boundary failure: job 8791583

Aurora debug job `8791583` tested source commit
`33b2b6ecd0a7a8cdee0d2689a483fabf8b934b54` on node `x4311c3s5b0n0`.
PBS charged `CompactBinaryMerger`, used one node for `00:20:19`, and recorded
exit status 143. The executable SHA-256 is
`25f1f1ea3e6bc46a2899ae883ba227fcf9f5b43303fa63db5194a16ca20e15c7`.

The build used IntelLLVM 2025.3.2, Kokkos 4.7.2, MPI, SYCL, PVC architecture,
and `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON`. Twelve ranks mapped to distinct
Level Zero PVC affinities `0.0` through `5.1`.

The source moves the exact matched q=1 metric and gauge boundary identities to
compile-time host dispatch. The compiler emits no spill warning for either
exact specialization; the approximately 122-Real warning remains only for the
unselected non-exact analytic projection specialization. Local ASan, UBSan,
and Kokkos-bounds validation had completed all four RK stages with the exact
specializations.

The PVC result nevertheless repeats the post-prolongation failure. All twelve
ranks completed RHS zeroing, Psi, separate gauge-driver, primary source/Pi,
standard-Phi, gamma2, dissipation, complete RHS, RK update, restriction, MPI
send/receive, and the synchronous post-prolongation fence. The next launch is
the exact projected-trumpet metric boundary. Before its stage-one completion
fence, multiple tiles reported distinct level-1 PDE `NotPresent` write pages
with `banned: 1`; rank 2 died from signal 6 and rank 1 was terminated. Kokkos
reported no bounds violation. Metric/gauge-boundary and `NewTimeStep` fence
counts in the compact summary are from cycle-zero initialization; they do not
show stage-one completion. The only history row remains finite at `t=0`.

This falsifies the specific hypothesis that the unused general projection's
private spills caused the observed stage-one fault. It does not distinguish a
boundary-view lifetime error, a GPU-aware MPI/device-page lifecycle problem,
or a lower-level compiler/runtime defect. The full 1.5 MiB build log remains
only on Aurora with SHA-256
`f97599177ccefd2131b0c47a373aec0cfa44e422c05a475d31b96bf3618ea9f7`.

Aurora campaign location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_phase8_exact_bc_20260830_33b2b6ec_v1`

This is a failed portability gate with no positive-time evolution, stability,
convergence, or production-readiness evidence. No follow-on run was launched.

# Phase-8 Aurora PVC split-gauge bounds failure: job 8791548

Aurora debug job `8791548` tested source commit
`cab2434170f12505b8d8400544845ff87fe31992` on node `x4705c2s1b0n0`.
PBS charged `CompactBinaryMerger`, used one node for `00:19:43`, and recorded
exit status 143. The executable SHA-256 is
`142a79c81f2e9c772eb365805e31edb9ca72ce4ba9d956394adad72408ed5366`.

The build used the Aurora IntelLLVM 2025.3.2 environment, Kokkos 4.7.2, MPI,
SYCL, PVC architecture, and `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON`. Twelve
ranks reported distinct Level Zero PVC affinities `0.0` through `5.1`.

The separate analytic gauge-driver kernel materially moved the synchronous
failure boundary. All twelve ranks completed RHS zeroing, the Psi RHS, the
gauge-driver RHS, the primary Pi/source RHS, standard-Phi RHS, gamma2 damping,
dissipation, the complete RHS fence, RK update, restriction, send, receive,
and prolongation. The next task is the projected-trumpet physical boundary.
No rank completed its metric-boundary fence before Level Zero reported a
level-1 PDE `NotPresent` GPU write fault at `0xff00000efc864000`. Rank 3 died
from signal 6 and rank 2 was terminated with signal 15. Kokkos emitted no
bounds violation. The only history row remains the finite `t=0` row.

The synchronous post-prolongation fence confines the observed fault to the
next projected metric-boundary launch, rather than the completed RHS or
communication kernels. The exact matched q=1 boundary currently selects a
small runtime branch inside a kernel that also contains the unused general
projection machinery; the compiler reports about 122 spilled Reals for that
analytic metric-boundary kernel. This is a portability diagnosis, not a
scientific evolution result and not proof of a continuum instability.

The full 1.5 MiB compiler log remains only on Aurora. Its SHA-256 is
`d9b425d81cb47e12eb3da61ec72b8d6089e34c9a14034b2c70460482fce3cdf6`.
The compact files here include the complete 40 KiB run log, rank-tagged fence
trace, finite cycle-zero histories, mapping, configuration, provenance, and
remote manifest.

Aurora campaign location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_phase8_split_gauge_20260830_cab24341_v1`

This is a failed PVC portability gate. It provides no positive-time,
stability, convergence, or production-readiness evidence. No follow-on
evolution was launched from this result.

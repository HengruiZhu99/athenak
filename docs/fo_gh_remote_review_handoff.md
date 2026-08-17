# Remote-agent handoff: vacuum FO-GH puncture driver

## Review target

Review branch `codex/fo-gh-puncture-driver-20260817` against base
`24dd527514a3b031d151ca8d3f2679e998a91b3d`.  Commit
`1e4c62ee25dd443334f8ddab1a27ad4d697e21d7` is the source state used for the
first captured Perlmutter campaign.  Commit
`3ec9c3bd326f22c7dedf792572876f4c2a8683a1` was used for the later current-source
preflight and doubled-domain controls.  The current worktree contains a newer
continuum correction described below; all Perlmutter puncture evolution
evidence predates it.

Do not work on fluid coupling.  Do not add Kerr-Schild or apparent-horizon
scope.  Do not infer production qualification from the passing preflight.

## What is implemented

- standalone 63-variable vacuum FO-GH module under `src/fo_gh/`;
- generic four-dimensional symmetric tensors and a 30-DOF mixed tensor;
- fixed regularized continuum RHS and independent hyperbolic gauge driver;
- compatible `Q`, `X`, `a`, and `B` evolution using the production derivative;
- 2/4/6 centered derivatives, matching-order sign-aware `Lx` shift advection,
  and modest configurable KO dissipation;
- exact identical-Z4c one-puncture data with the puncture between cells;
- diagnostic-only lapse excision equivalent to Z4c's `chi=0.0625` mask;
- ADM adapter, characteristic timestep, restart/load-balance support;
- static refinement and dynamic-regrid gradient repair;
- exact/robust Minkowski, wave, algebra, geometry, RHS, puncture, restart, and
  AMR tests.
- history reductions for the four curl constraints, determinant/trace
  constraints, gauge residuals, and separately normalized fixed-radius
  `H/M/GH/reduction+curl` families; current momentum norms use the Z4c-style
  physical inverse-metric contraction under the same `alpha>=0.25` mask.

## Review priorities

1. Review the correction that constructs the symmetric twice-raised
   `Atilde^{ij}` only once per stored component and uses it, rather than
   `Atilde^i_j`, in the `Atilde^{ik} a_k` and `Atilde^{ik} X_k` Lambda terms.
2. Review the new independent non-diagonal metric-jet Ricci, `D_i c^i`,
   Hamiltonian, momentum, and full nonlinear RHS regressions.  The latter
   directly covers the `Atilde` TF and Lambda Lie-index terms against the fixed
   equations.
3. Check the two-pass stencil/ghost contract and whether KO treatment preserves
   compatibility at physical and coarse/fine boundaries.
4. Rerun the corrected source on GPU; the earlier robust-advection/history
   preflight predates the non-diagonal RHS correction.
5. Preserve fixed-region diagnostics alongside the lapse mask.  The completed
   same-spacing small/doubled-box comparison shows that moving the face from
   `4M` to `8M` does not alter the central histories through `5M`; the large
   masked jumps instead coincide with resolution-dependent mask-volume jumps.
6. Before long continuation, repeat a bounded `[-8M,8M]^3` `N=32,48,64`
   ladder with the corrected source and require exterior GH convergence.

## Reproducible source state

The intended source is the Git tree, including committed Kokkos gitlink
`6739bc623081648af9e752b616d9671527922cbf`.  The local checkout has unrelated
dirty paths (`kokkos`, `cactus_einsteintoolkitanalysis`, `twopuncturesc`) that
were deliberately not staged.  Do not use the dirty local Kokkos checkout as
source provenance.

The Perlmutter source archive SHA-256 is
`9d813ef243812c08f6a785ff976f24cc0c96209ba01715f08d4f348d3048c357`.
The built executable SHA-256 is
`11fb6955d027f50e69610e3a6da7b4656c1ff6536d41d1884920da8439acf5a2`.

## Evidence map

- `docs/fo_gh_puncture_formulation.md`: fixed equations and code mapping.
- `docs/fo_gh_puncture_validation.md`: results, failures, and interpretation.
- `docs/fo_gh_artifacts/perlmutter_20260817/evidence/provenance.txt`: allocation,
  commit, submodule, modules, node, and GPU.
- `.../evidence/cmake_key_values.txt`: CUDA/AMPERE80/compiler configuration.
- `.../evidence/gpu_dmon.txt`: actual A100 runtime telemetry.
- `.../evidence/pconv_summary.txt`: lapse-masked puncture convergence.
- `.../evidence/long_u_t1_summary.txt`: three-resolution `1M` gate.
- `.../evidence/long_u_t5_summary.txt` and `long_u_t5_traces.txt`: failed `5M`
  convergence gate and onset traces.
- `.../runs/`: raw history and checkpoint text files.
- `docs/fo_gh_artifacts/local_boundary_control/`: doubled-domain coarse control.
- `docs/fo_gh_artifacts/local_parameter_sweep/`: small-box diagnostic sweep.
- `docs/fo_gh_artifacts/perlmutter_20260817_current/README.md`: later-source
  A100 preflight, equal-spacing domain control, and exterior-norm summary.

The large restart files remain in
`/pscratch/sd/h/hzhu/fo-gh-puncture-20260817.L0vwO0`; the repository contains
their SHA-256 manifest rather than duplicating them.

## Exact current scientific status

On pre-correction source, uniform smooth waves converge at order `3.91--3.93`
and the real-SMR result is only order `1.607`.  Exact and robust Minkowski,
compatible gradients, dynamic regrid repair, and bitwise restart continuity
passed on one A100.

The pre-correction identical-data puncture is finite through `5M`.  Equal-dx
small/doubled-box central histories agree through `5M`, so the central behavior
is not caused by boundary arrival on that interval.  Moving lapse-mask volume
and stalled exterior GH convergence prevent a stability/convergence claim.
The corrected RHS has only focused local evidence.  No run beyond `5M`
and no prior GPU result should be used to qualify it.

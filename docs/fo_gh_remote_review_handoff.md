# Remote-agent handoff: vacuum FO-GH puncture driver

## Review target

Review branch `codex/fo-gh-puncture-driver-20260817` against base
`24dd527514a3b031d151ca8d3f2679e998a91b3d`.  The source implementation under
review ends at `1e4c62ee25dd443334f8ddab1a27ad4d697e21d7`; the later commit adds
only this documentation and captured evidence.

Do not work on fluid coupling.  Do not add Kerr-Schild or apparent-horizon
scope.  Do not infer production qualification from the passing preflight.

## What is implemented

- standalone 63-variable vacuum FO-GH module under `src/fo_gh/`;
- generic four-dimensional symmetric tensors and a 30-DOF mixed tensor;
- fixed regularized continuum RHS and independent hyperbolic gauge driver;
- compatible `Q`, `X`, `a`, and `B` evolution using the production derivative;
- 2/4/6 finite differences, modest configurable KO dissipation;
- exact identical-Z4c one-puncture data with the puncture between cells;
- diagnostic-only lapse excision equivalent to Z4c's `chi=0.0625` mask;
- ADM adapter, characteristic timestep, restart/load-balance support;
- static refinement and dynamic-regrid gradient repair;
- exact/robust Minkowski, wave, algebra, geometry, RHS, puncture, restart, and
  AMR tests.

## Review priorities

1. Audit `fo_gh_geometry.hpp` and `fo_gh_rhs.hpp` index by index against
   `docs/fo_gh_puncture_formulation.md`; do not replace the fixed equations.
2. Verify `DivergenceC`, conformal Ricci quadratic terms, the `Atilde` TF terms,
   and the Lambda Lie-index term independently rather than trusting the current
   pointwise unit test.
3. Check the two-pass stencil/ghost contract and whether KO treatment preserves
   compatibility at physical and coarse/fine boundaries.
4. Treat the `[-4M,4M]^3` loss near the `4M` half-crossing time as potentially
   boundary-driven.  The single doubled-box coarse control is not decisive.
5. Before long continuation, run a `[-8M,8M]^3` `N=32,48,64` ladder with
   fixed central-region histories and measured characteristic arrival.

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

The large restart files remain in
`/pscratch/sd/h/hzhu/fo-gh-puncture-20260817.L0vwO0`; the repository contains
their SHA-256 manifest rather than duplicating them.

## Exact current scientific status

Uniform smooth waves converge at order `3.91--3.93`.  The real-SMR wave result
is only order `1.607`.  Exact and robust Minkowski, compatible gradients,
dynamic regrid repair, and bitwise restart continuity pass on one A100.

The identical-data puncture is finite through `5M`, but its lapse-masked
constraints cease to improve with resolution on the current small box.  This
is not a long-stability result.  No run beyond `5M` should be used for a claim
until the boundary/characteristic-arrival experiment above is complete.

# Coupled Cartesian-core / spherical-shell reproduction

The new hyperboloidal equations now evolve coupled 3D fields on a Cartesian core and six mapped patches in each of three spherical-shell layers. The CPU control and the new distributed MPI/SYCL research runner remain separate from AthenaK's production task graph. **A stable, accurate wormhole-to-trumpet evolution through 100 M has not been demonstrated. Production hyperboloidal evolution remains disabled.**

Branch: `codex/hyperbolic-core-shell`, based on `4b4cef316411932a12cbbbf2bcbb110a70f1df90` (`codex/hyperbolic-shift-scri-3d`). The equation/control donor is `c0ae7472a7aecc92cc72312c33d4cdc417a26291` (`codex/hyperboloidal-hyperbolic-shift`). No standard-Z4c validation campaign was run.

## Aurora work resumed on 2026-10-28

The user authorized one Aurora debug node at a time, using the `CompactBinaryMerger` allocation, all 12 MPI ranks and all 12 GPU tiles. The campaign lives under `/home/hzhu/hyperbolic-core-shell-20261028`, with a 5,000,000,000-byte campaign storage cap, explicit output/checkpoint reserves, and retained small diagnostics. Builds use MPI and SYCL with `make -j64`. Long-time stability of both Minkowski and the isotropic wormhole comes before a convergence campaign.

The standalone `hyperbolic_shift_3d_mpi` runner distributes volume cells and actual-scri surface rows among ranks. Persistent Kokkos kernels evaluate the native volume equations, mapped derivatives, transport, KO dissipation, surface tensor limits, dependent gauge closure and RK stages. GPU-aware MPI exchanges global active stages and surface data; physics is evaluated on owned cells/rows. This first implementation still replicates global stage storage and mesh plans, so it is not a scalable production AMR implementation.

Local Kokkos-Serial verification establishes bitwise agreement of volume, scri and both compensation arrays with the CPU control; one, two and twelve ranks agree, and uninterrupted/restarted short evolutions agree. These are local correctness tests, not yet Aurora GPU or long-time stability results. See [local evidence](mpi-full-device-local-v2/scope.json) and the [Aurora launcher documentation](../../../tst/hyperbolic_shift_3d/aurora/README.md).

The existing fast-flow horizon finder now accepts physical Cartesian geometry reconstructed from hyperboloidal fields, with patch-aware sampling. It has analytic Schwarzschild, displaced-horizon and actual puncture-grid tests. Expansion residuals and finite-resolution acceptance thresholds are recorded explicitly; an unfound horizon leaves exterior norms unavailable. See [horizon validation](../horizon.md).

The first Aurora allocation, job `8981303`, verified twelve distinct GPU tiles and the component arithmetic check measured a maximum scaled CPU/SYCL difference of `3.40e-14`. The preflight rejected the job because the test binary lacked its required backend identifier: an older object survived source extraction with preserved timestamps. No physical evolution ran. A fresh build directory is required for each exported scientific snapshot; a source hash alone does not prove that an incrementally reused object was recompiled. The [recorded outcome](aurora/preflight-v1/assessment.json) remains a preflight rejection.

The original J-disabled, bulk-mu=4 1D control is confirmed to have completed coordinate `33.3333984375 = 100.0001953125 M`. Its N513/N1025 continuation chains and frozen-source identities were checked. A [portable N513 time series](original-1d-reference/history.csv) and [provenance](original-1d-reference/provenance.json) preserve the matching horizon/exterior-constraint reference. Late gauge settling and uniform puncture convergence were not established by that original control.

## Archived pause point

At the previous user-requested pause, no evolution processes remained. Changes were local and uncommitted on `codex/hyperbolic-core-shell` and had not been pushed. See the [paused overview](paused-overview.png), [recorded summary](paused-summary.json), [duration comparison](duration-comparison-v2/summary.json), and [three-resolution pulse comparison](pulse-comparison-v3/summary.json).

The refined n=16 run completed t=0.1=0.3 M: exterior H/M radial RMS were 0.05463/0.10928, the spherical horizon mass error was 4.98e-5, and maximum pole residual was 7.69e-3. The 1D H/M RMS at that time were 2.55e-4/4.84e-4. There is short-time improvement under refinement, but no qualified 100 M reproduction. Pulse lapse/H errors improve through n=24; momentum and Theta do not all converge monotonically.

An independent audit found an 8.64e-9 null defect between surface nodes, despite 1e-15 nodal nullness. The final code now reconstructs the shift from the interpolated live geometry at every boundary evaluation. Its actual ghost-lift null test gives 1.11e-15. **No evolution rerun after this last correction was completed before the pause.** The plots are from the preceding trajectory implementation.

The earlier n=16 duration attempt stopped at t=0.0475; the n=20 attempt stopped at t=0.04365. They are bounded outcomes, not numerical failures or completed t=0.1 runs. The last n=24 pulse and n=20 bounded run reported wall times of 1555 s and 1286 s despite the intended 410 s external limit. The cause has not been established; those runs do not certify compliance with the timing protocol.

## What is implemented

`src/z4c/hyperboloidal/hyperbolic_composite.hpp` evolves all six metric components, six curvature components, chi, physical P and Theta, independent Lambda, lapse, telegrapher B, shift and its independent D auxiliary. Components are global Cartesian components throughout. Initial data are spherically symmetric; evolution does not average over angles, reconstruct a radial solution, freeze the geometry, or substitute metric-built Lambda. The optional three-component adaptive source J is available explicitly; the present baseline comparisons select the original hyperbolic driver with J disabled.

The host driver has coupled RK4 stages, compensation, fourth-order centered geometric derivatives, fourth-order biased transport, sixth-degree boundary ghost interpolation, and block-local KO8 dissipation in `-V^-1 B^T W B` form. Dissipation's weighted energy contribution is nonpositive. The Cartesian core is cell centered with an even number of cells: the puncture is not a grid point. There is no puncture excision, inner boundary condition, lapse floor, or spherical parity operation. This direct-variable discretization differs from the 1D logarithmic puncture chart; its late-time adequacy must be measured.

The grid currently has core half-width 0.125 and shell radii 0.45, 0.75, and 1. The number of cells per axis per block is varied. The four original copied mesh/scri headers came from `6d54b2ef7154102325d83ef4faa3917e52b28df6`; `imported-infrastructure.json` records their **original**, not current, hashes. No elliptic evolution engine was imported. The derivative and tensor-scri adapters have subsequently been changed and tested on this branch.

## Scri boundary

An explicit atlas stores fields at actual Omega=0, separate from cell-centered volume fields. At every stage, lapse is one, and the coordinate tangential shift is zero. With outward coordinate covector n=x, the remaining shift is reconstructed as `beta=-sqrt(chi*g^{nn})*x`. The D trace is eliminated using the differentiated null relation and the same live geometric boundary rate. Independent Lambda and the geometric values are not projected. The restoring-term limit is differentiated from the actual shift deviation.

Geometric boundary rates use the full tensor pole's radial derivative, including angular derivatives in its underlying jets. Weak numerator and regularity penalties follow the spherical donor's coefficients. This is an implemented boundary prescription, **not a proof of a stable general 3D boundary problem**. Nullness, dependent-driver compatibility, pole numerators, and the discarded lapse-clock residual are recorded separately. Small null error cannot excuse large geometric pole errors.

The discrete closure audit checks nonspherical tensor poles against the native bulk equations, their derivatives against independent finite differences, boundary traces against ghost reconstruction, and spherical reduction of the full coordinate-sphere expansion diagnostic. It also checks serial/threaded RHS identity, a coupled Minkowski step, and rejected-step preservation. The existing mapped-mode and hyperbolic point/symbol tests pass. The known principal-symbol collision remains a diagnostic; the bulk coefficient has not been changed to avoid it.

## Initial results

Minkowski equilibrium completed t=0.05 on the 19-block n=6 grid. Its maximum scri null residual was 1.0e-15 and maximum pole numerator 1.7e-13. A finite lapse pulse is evolved and compared with the unchanged 1D pulse driver, separately from the equilibrium test.

For the wormhole, M=1/3 and the first comparison is t=0.02=0.06 M. The initial Schwarzschild slice, isotropic radial map, lapse, and integrated initial shift are those of the donor; this is not ordinary time-symmetric puncture data substituted into a different outer slicing. All four resolutions completed:

| Cells per block axis | Active cells | Exterior H RMS | Exterior radial M RMS | Max scri pole | Max null error |
|---:|---:|---:|---:|---:|---:|
| 6 | 4,104 | 0.2240 | 0.6591 | 8.03e-2 | 1.23e-15 |
| 8 | 9,728 | 0.2560 | 0.3115 | 1.33e-2 | 1.12e-15 |
| 12 | 32,832 | 0.1217 | 0.1187 | 8.00e-5 | 1.12e-15 |
| 16 | 77,824 | 0.0673 | 0.0609 | 5.11e-7 | 1.12e-15 |

The H norm is not monotonic on the two coarsest grids. These grids are not a demonstrated fourth-order convergence regime. The n=16 coordinate-sphere horizon mass estimate has fractional error -1.12e-5, but it is a spherical diagnostic, not a general apparent-horizon finder. Angular spread is retained in every comparison. Exterior norms use the same compact-coordinate volume measure and the contemporaneous 1D outer-horizon cutoff.

[Short-time comparison](comparison-v2/comparison.png), [numerical summary](comparison-v2/summary.json), and [timestep control](time-refinement.json). Halving the n=12 timestep from 0.0005 to 0.00025 changes the volume-weighted lapse RMS by 1.41e-10 and vector-shift RMS by 2.83e-9. Spatial errors dominate this comparison.

The first n=12 duration run completed t=0.1=0.3 M, but its maximum pole residual reached 4.06e-2. It is not qualified for extrapolation to 100 M. A slower n=16 duration trial and subsequent performance changes are recorded separately; bounded outcomes must not be relabeled as completed evolutions.

The first ray plots used a one-sided active-only polynomial and exaggerated seam errors. `comparison-v1` is retained as a superseded diagnostic. `comparison-v2` reanalyzes the same saved states with the live ghost reconstruction. Volume constraint measurements and evolution states did not change.

## Reproduce

Build only the new hyperboloidal targets. Set `Python3_EXECUTABLE` to a Python with NumPy installed.

```sh
cmake -S . -B build-hyperbolic-core-shell \
  -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/path/to/python
cmake --build build-hyperbolic-core-shell --target \
  hyperbolic_shift_3d_native hyperbolic_shift_3d_coupled \
  hyperbolic_shift_3d_tensor_scri hyperbolic_shift_3d_mapped \
  hyperbolic_shift_3d_reanalyze -j2
python tst/hyperbolic_shift_3d/freeze_build.py build-hyperbolic-core-shell
ctest --test-dir build-hyperbolic-core-shell -R '^hyperbolic_shift_3d_' --output-on-failure
python tst/hyperbolic_shift_3d/coupled_trial.py \
  --build build-hyperbolic-core-shell --output output/new-puncture-n12 \
  --n 12 --kind puncture --target .02 --dt .0005 --workers 4
```

Each trial uses a fresh directory and immutable executable, records source and executable hashes, and has an external 410-second bound. No failed/bounded state is used as a continuation. Keep at most two evolution processes running. `baseline.py` makes a bounded fresh 1D control; `compare_coupled.py` makes the matched-time plot and JSON norms. `reanalyze.cpp` only reads saved volume and surface fields and does not advance the equations.

## Remaining work

1. Establish sustained convergence of the exterior solution, scri regularity and horizon diagnostics before a long wormhole-to-trumpet campaign. The 100 M result from 1D has not transferred to 3D.
2. Qualify the new adaptive source separately from the original-driver control, including angular perturbations and late-time settling.
3. Qualify the new standalone MPI/SYCL runner on Aurora, then connect the numerical path to the production stage graph. Distributed native physics and persistent device storage are implemented; production task-graph integration is pending.
4. Establish GPU checkpoint/restart equivalence. The new binary checkpoint preserves all volume/surface fields and RK compensation and rejects failed states; local rank/restart equivalence tests pass. Bound source/binary/input manifests and scheduler admission guards are implemented.
5. Connect Cartesian AMR to stage-consistent transfer of all fields and auxiliaries, rebuild the mesh/scri plans, and verify coarse/fine interfaces, refinement changes and moving-puncture crossings. The copied mesh descriptor can refine Cartesian leaves, but that is not an integrated evolution/AMR implementation.
6. Add genuinely nonspherical evolutions, angular boundary perturbations and seam-crossing tests. The generalized fast-flow horizon finder is implemented and locally tested; general dynamical 3D use remains to be qualified. The initial reproduction evolves full 3D fields but does not validate arbitrary nonspherical data.

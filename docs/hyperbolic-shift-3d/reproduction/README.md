# Coupled Cartesian-core / spherical-shell reproduction

The new hyperboloidal equations now evolve coupled 3D fields on a Cartesian core and six mapped patches in each of three spherical-shell layers. The CPU control and the new distributed MPI/SYCL research runner remain separate from AthenaK's production task graph. **A stable, accurate wormhole-to-trumpet evolution through 100 M has not been demonstrated. Production hyperboloidal evolution remains disabled.**

Branch: `codex/hyperbolic-core-shell`, based on `4b4cef316411932a12cbbbf2bcbb110a70f1df90` (`codex/hyperbolic-shift-scri-3d`). The equation/control donor is `c0ae7472a7aecc92cc72312c33d4cdc417a26291` (`codex/hyperboloidal-hyperbolic-shift`). No standard-Z4c validation campaign was run.

## Aurora work resumed on 2026-10-28

The user authorized one Aurora debug node at a time, using the `CompactBinaryMerger` allocation, all 12 MPI ranks and all 12 GPU tiles. The campaign lives under `/home/hzhu/hyperbolic-core-shell-20261028`, with a 5,000,000,000-byte campaign storage cap, explicit output/checkpoint reserves, and retained small diagnostics. Builds use MPI and SYCL with `make -j64`. Long-time stability of both Minkowski and the isotropic wormhole comes before a convergence campaign.

The standalone `hyperbolic_shift_3d_mpi` runner distributes volume cells and actual-scri surface rows among ranks. Persistent Kokkos kernels evaluate the native volume equations, mapped derivatives, transport, KO dissipation, surface tensor limits, dependent gauge closure and RK stages. GPU-aware MPI exchanges global active stages and surface data; physics is evaluated on owned cells/rows. This first implementation still replicates global stage storage and mesh plans, so it is not a scalable production AMR implementation.

Local Kokkos-Serial verification establishes bitwise agreement of volume, scri and both compensation arrays with the CPU control; one, two and twelve ranks agree, and uninterrupted/restarted short evolutions agree. The separate Aurora preflight below now supplies actual GPU component and short-restart evidence; neither establishes long-time stability. See [local evidence](mpi-full-device-local-v2/scope.json) and the [Aurora launcher documentation](../../../tst/hyperbolic_shift_3d/aurora/README.md).

The existing fast-flow horizon finder now accepts physical Cartesian geometry reconstructed from hyperboloidal fields, with patch-aware sampling. It has analytic Schwarzschild, displaced-horizon and actual puncture-grid tests. Expansion residuals and finite-resolution acceptance thresholds are recorded explicitly; an unfound horizon leaves exterior norms unavailable. See [horizon validation](../horizon.md).

The first Aurora allocation, job `8981303`, verified twelve distinct GPU tiles and the component arithmetic check measured a maximum scaled CPU/SYCL difference of `3.40e-14`. The preflight rejected the job because the test binary lacked its required backend identifier: an older object survived source extraction with preserved timestamps. No physical evolution ran. A fresh build directory is required for each exported scientific snapshot; a source hash alone does not prove that an incrementally reused object was recompiled. The [recorded outcome](aurora/preflight-v1/assessment.json) remains a preflight rejection.

The fresh `build-sycl-mpi-v4` preflight, job `8981414`, passed all nine cases on 12 distinct GPU tiles using frozen source equivalent to commit `113aaff783008204e1ff8cbf2fb8bb5f2af2204c`. Native RHS CPU/SYCL scaled differences are at most `3.40e-14`, and scri-rate differences `7.82e-14`. Direct four-step and restarted two-plus-two-step pulse checkpoints are bitwise identical, including both compensation arrays. Minkowski and the isotropic puncture completed only `t=0.002` at `n=16`, `dt=0.0005`; the puncture duration is **0.006 M**. Surface/ghost null residuals remain at most `1.23e-15`. Actual FastFlow finds the puncture horizon with expansion RMS `4.50e-4` under the recorded coarse-grid tolerance `1e-3`. All 58 worker-bound artifact hashes and 94 report checks were independently verified. See the [portable preflight assessment](aurora/preflight-v2/README.md). These are preflight results, not sustained-stability or convergence results.

The longer original-scheme n16 runs failed: [Minkowski](aurora/stability-v1/README.md)
at coordinate t=0.4895 and the [puncture](aurora/stability-puncture-v1/README.md)
at t=0.369=1.107 M, both with nonpositive interpolated scri geometry. Null closure
remained near roundoff while field and constraint errors grew. The Minkowski
failure is strongly concentrated at outer-shell face edges/corners; this
localization does not by itself identify the unique unstable operator. The
[unchanged-executable n16/n24/n32 spatial and timestep controls](aurora/resolution-control-comparison-v1/README.md)
are now complete. Fitted growth rates increase from 74.49 to 116.35 to 153.87
per coordinate time under spatial refinement; halving the n16 timestep changes
the rate by less than one percent. All four controls fail the scientific
stability limits. Thus modest spatial refinement or a smaller timestep does
not cure this instability. Larger blocks improve measured cell throughput,
but global communication costs grow strongly. These are stability controls,
not evidence of physical convergence.

The puncture horizon rejection has a separately demonstrated measurement
limitation. On its fixed unadmitted t=0.369 state, lmax6/ntheta12 stalls at
expansion RMS 0.00574 after both 300 and 1000 iterations, while lmax12/ntheta24
finds a horizon at RMS 0.000221 with the original 0.001 tolerance. See the
[fixed-snapshot audit](horizon-angular-audit-v1/interpretation.md). This does not
admit the failed evolution or establish spatial accuracy.

Two optional repairs have controlled comparison evidence:
[standard final-stage tensor projection](../algebraic-projection-audit.md) and
[cross-face scri angular reconstruction](../scri-angular-exchange.md). Their
checkpoint schemas are distinct from the original control. Projection-only
Aurora preflight job 8982210 passed all twelve component, benchmark and four-step
checks; no long-time stability claim follows. The staged scri sampling
microbenchmark reduced mean Evaluate time from 47.64 ms to 3.72 ms per rank
(12.81x), with matching per-rank output norms. This is a kernel microbenchmark,
not a measured whole-evolution speedup. The subsequent
[projection-only duration test](aurora/stability-projected-minkowski-v1/README.md)
failed at t=.485 with a stored scri RK-stage metric becoming indefinite;
accepted algebraic residuals remained near roundoff. Its measured state-growth
rate was 75.81, so final algebraic repair did not suppress the growing mode.

The [cross-face V3 preflight](aurora/preflight-angular-v1/README.md) passed all
twelve GPU component, four-step, exact-restart and communication checks.
Its [fresh duration control](aurora/stability-angular-minkowski-v1/README.md)
then failed at accepted t=.5205 with nonpositive interpolated scri geometry.
The last fresh diagnostic at t=.5 has state deviation 29.76 and H/M RMS
.07128/1.92019, while null closure remains near 1e-15. Its state-growth rate
is 72.44 on the same t=.15–.35 fit interval. Cross-face derivative
jets alone therefore do not establish stability. Owner-face ghost-value
interpolation and the original regularity penalties are unchanged, and no
explicit angular dissipation acts at scri. The full coupled discrete operator
and the remaining boundary interpolation need further investigation.
The recorded all-direction shift/lapse speed-gap diagnostic remains positive
(minimum .54185) in this run; it does not demonstrate the earlier bulk
tangential-speed collision. This diagnostic alone is not a complete
hyperbolicity or boundary-stability proof.

An optional V4 candidate adds degree-seven angular dissipation on actual
scri nodes, using `Q=-epsilon*n*M^-1*B^T*M*B`. The
[standalone operator study](scri-dissipation-prototype-v1/README.md) checks
polynomial consistency, energy, cross-face support and reference-sphere areas.
The native plan uses twice-orthogonalized QR and paired donor/anchor transpose
contributions; its conservative incidence-based spectral bound is larger than
the prototype's combined-diagonal bound. The added operator is dissipative in
its fixed area-weighted norm, which is not a stability proof for the coupled
equations, nonlinear projection or signed ghost interpolation.

The optional `hyperbolic_shift_3d_mpi_dissipative` target applies Q to the 21
free surface fields before reconstructing dependent D. Lapse, shift, D and J
are not independently filtered. It retains the V3 cross-face jets and accepted
tensor projection; the original V1/V2/V3 executables remain available. Its
default strength is epsilon=2, with an optional thirteenth positional argument
after the horizon angular settings. Epsilon=0 provides a disabled control.
The V4 checkpoint adds a 16-byte epsilon/degree/reserved configuration after
the original 72-byte header and rejects cross-scheme or different-strength
restarts. An isolated-Q RK4 guard requires
`dt*epsilon*n*spectral_upper_bound <= 2.7`; this guard does not admit a coupled
timestep. Local native operator and closure tests pass, including nonspherical
rates, exact constant preservation, weighted energy, CPU/device-interface
parity and twelve ownership partitions. Fifteen bounded native MPI cases
confirm a bitwise V3/zero-epsilon payload match, one-rank direct versus
two-rank restarted equality of all four checkpoint arrays, and configuration,
admission and CFL rejection. These local checks use Kokkos Serial.
The [native verification package](angular-dissipation-native-v1/README.md)
retains the final source bindings, small raw outputs and checkpoint hashes.

The subsequent [V4 SYCL preflight](aurora/preflight-dissipative-v1/README.md),
job 8983792, passed all fifteen cases on twelve GPU tiles. Independent review
verified 221 checks, 131 artifacts, all 84 component rank logs and the complete
frozen source/build identity. The four-step epsilon-two pulse restart is
bitwise identical to its direct run, and V4 epsilon zero reproduces the V3
four-array payload exactly. Minkowski state deviation is 1.01e-14 at t=.002;
the puncture reaches only .006 M, with null residual <=1.23e-15 and horizon
expansion RMS .000325. This preflight admitted epsilon two for a
[bounded fresh Minkowski duration comparison](aurora/stability-dissipative-minkowski-v1/protocol.md).
The [V4 duration test](aurora/stability-dissipative-minkowski-v1/README.md)
subsequently failed at t=.5225 with nonpositive interpolated scri geometry.
Its measured growth rate, 72.195, is only 0.332% below V3's, and the same rapid
instability persists. At the last diagnostic t=.5, state deviation is 21.28
and H/M RMS are .05655/1.65426, despite null closure near 1e-15. An exact
replay finds positive metrics at all stored scri nodes but an indefinite
ghost interpolation; this identifies the immediate rejection, not the initial
growth's cause. V4 is not a stable-duration or 100 M result.

Three [fresh diagnostic captures](aurora/linear-window-dissipative-v2/acquisition/README.md)
at t=.25/.30/.35 completed with the same V4 binary and parameters. Their
acquisition checks passed, but every snapshot violates the original state-error
limit and the last also violates the momentum limit. They are retained to
analyze the growing direction, with no scientific restart admission. The
[saved-state angular-response study](scri-collar-response-v1/README.md) does
not justify extending the current angular filter into a volume collar. Large
superseded V4 payloads were [archived and purged with verified hashes](aurora/v4-retention-v1/README.md);
the three diagnostic captures remain available.
Their [early error profiles](early-direction-figure-v1/README.md) show a nearly
unchanged radial surface/interior mismatch. The independently reviewed
[normal-discretization audit](scri-normal-discretization-audit-v1/README.md)
identifies differences from the successful 1D endpoint operator: radial node
layout, sampled-numerator versus direct-jet differentiation, and outer ghost
feedback. It proposes an isolated control; none of those differences has yet
been established as the unique instability cause or repaired by that audit.

The [native directional replay](scri-directional-replay-v1/README.md) reproduces
the measured early growth with the full coupled RHS: the projected rate is
72.9635 and the relative shape residual is 0.0140 on the t=.30 error direction.
The surface normal-pole term contributes +110.981 to the projected surface
action, while the regular term contributes -53.4475. These are contributions
to one directional action, not independent eigenvalues. Replacing only the
normal derivative by the derivative of native pole numerators sampled on the
same six radial nodes changes the held-other-terms surface action from
+72.9647 to -27.0231. This motivates a controlled experimental closure; it does
not establish stability of a trajectory.

An [optional CPU sampled-numerator closure](scri-sampled-normal-cpu-v1/README.md)
now implements that control, preserving independent Lambda and Theta, the weak
penalties, angular operators and moving-normal null closure. It fills provisional
geometric ghosts, computes the final geometric rate and dependent D, then
refreshes ghosts before the bulk RHS. Ordering tests detect a deliberately stale
D ghost field; surface null error remains 1.11e-15. The default analytic-jet
path is bitwise identical to the frozen V4 control across six fixtures.
The [manufactured accuracy study](scri-sampled-accuracy-v1/README.md) finds
roughly third-order complete sampled-closure accuracy over n8/12/16/24,
compared with approximately fourth order for the analytic-jet closure. The
candidate is consistent but does not retain the original boundary order.
These checks alone provide no evolution, restart, GPU or stability admission.

The subsequent [complete coupled candidate action](scri-candidate-action-v1/README.md)
includes the new dependent D and refreshed volume ghosts. On the same t=.30
direction, the whole-state projected rate changes from +72.9635 to -22.3588;
the surface rate is -27.0231. The whole-state shape residual increases to
0.6888, so this is not a negative eigenvalue or a stable evolution. The volume
instantaneous action remains nearly unchanged. All nine default-path arrays
match the frozen V4 oracle bitwise, and candidate surface/ghost null residuals
stay at or below 1.11e-15. The subsequent candidate duration result is recorded below.

The optional V5 `hyperbolic_shift_3d_mpi_sampled_normal` target now provides
[device numerator sampling](sampled-normal-device-local-v1/README.md) and
[coupled MPI integration](scri-sampled-normal-mpi-v1/README.md). A separate halo
plan supplies the blocks owned by each rank's surface rows; it does not assume
those blocks match volume ownership. Native bulk jets produce five numerator
samples per surface row, then the endpoint derivative enters before the final D
trace and refreshed volume ghosts. The V5 checkpoint binds its normal scheme,
six-node layout and bulk derivative order in a distinct 104-byte header.
[Controller and reader checks](scri-sampled-normal-launcher-v1/README.md) prevent
older-closure evidence or checkpoints from qualifying this candidate.

Local Serial component tests give exact CPU/device-interface agreement for
numerators, surface rates and final D, including twelve simulated ownership
partitions. One-/two-rank four-step MPI checkpoints and a two-plus-two-step
restart agree bitwise, as does rebuilt V4 against its frozen executable.
These are short implementation checks, not actual GPU or sustained-stability
results. The manufactured third-order boundary accuracy remains a limitation.
The scientific source is frozen at `9d72318c14a2099b5aea118cca62e9fbd8773350`
for a fresh MPI/SYCL build and separate Aurora validation.

The [actual V5 twelve-tile preflight](aurora/preflight-sampled-normal-v1/README.md)
passed all 167 checks, including CPU/SYCL parity, exact direct/restarted pulse
checkpoints and four-step Minkowski/puncture evolutions. Its
[fresh Minkowski duration test](aurora/stability-sampled-normal-minkowski-v1/README.md)
then completed t=.6 without a metric exception but **failed stability**:
state error 7.20e-4 and momentum RMS 2.53e-6 exceed unchanged limits. Null
closure remains within 1.23e-15. Measured growth falls from V4's 72.195 to
41.842 per coordinate time, but remains positive. This supports a significant
normal-closure contribution without establishing a stable scheme. The V5
checkpoint is scientifically rejected despite its native completion bit;
the remaining coupled growing direction still needs diagnosis.

The [actual V5 direction diagnostic](scri-v5-direction-v1/README.md) finds a
full coupled projection +42.5680 with shape residual .01425, consistent with
the observed growth. Surface and volume rates agree. Curvature A carries
99.06% of its unweighted squared coordinate norm. In this direction the
sampled normal term contributes -50.84, while the numerator and regularity
penalties together contribute +118.28. Their native term sum reproduces the
full surface action to 4.11e-12 relative error. This identifies the coupled
A/metric regularity penalty as the next target; it does not justify stronger
penalties or claim that removing them would stabilize the scheme.

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

An explicit atlas stores fields at actual Omega=0, separate from cell-centered volume fields. At every stage, lapse is one, and the coordinate tangential shift is zero. With outward coordinate covector n=x, the remaining shift is reconstructed as `beta=-sqrt(chi*g^{nn})*x`. The D trace is eliminated using the differentiated null relation and the same live geometric boundary rate. In the original control, independent Lambda and the geometric values are not projected; the optional tensor-repair candidate still retains independent Lambda. The restoring-term limit is differentiated from the actual shift deviation.

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

1. Establish sustained low-resolution stability of both Minkowski and the isotropic wormhole before increasing resolution for convergence of the exterior solution, scri regularity and horizon diagnostics. The 100 M result from 1D has not transferred to 3D.
2. Qualify the new adaptive source separately from the original-driver control, including angular perturbations and late-time settling.
3. Extend the passed Aurora component/short-evolution preflight to sustained runs, then connect the numerical path to the production stage graph. Distributed native physics and persistent device storage are implemented; production task-graph integration is pending.
4. Extend the passed four-step GPU checkpoint/restart test to long trajectories and subsequent mesh/AMR changes. The binary checkpoint preserves all volume/surface fields and RK compensation and rejects failed states; local rank/restart tests and the bounded Aurora restart comparison pass. Bound source/binary/input manifests and scheduler admission guards are implemented.
5. Connect Cartesian AMR to stage-consistent transfer of all fields and auxiliaries, rebuild the mesh/scri plans, and verify coarse/fine interfaces, refinement changes and moving-puncture crossings. The copied mesh descriptor can refine Cartesian leaves, but that is not an integrated evolution/AMR implementation.
6. Add genuinely nonspherical evolutions, angular boundary perturbations and seam-crossing tests. The generalized fast-flow horizon finder is implemented and locally tested; general dynamical 3D use remains to be qualified. The initial reproduction evolves full 3D fields but does not validate arbitrary nonspherical data.

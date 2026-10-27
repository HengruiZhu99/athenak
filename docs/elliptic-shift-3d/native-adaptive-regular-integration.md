# Accepted native adaptive transaction: regular Minkowski qualification

The experimental native owner now supports an **explicit opt-in regular
Minkowski adaptive path**, including actual Driver refinement and coarsening.
Wormhole adaptive evolution, MPI/device, matter, tracker motion, cyclic zoom,
logical-shape changes and midstage transactions remain unsupported. Host-owned
shell/scri fields are still not native MeshBlocks. The scientific wormhole,
late-time lapse-front and moving-puncture milestones are not completed by this
mechanical/gauge-control phase.

## Transaction and authority

`MeshRefinement::RedistAndRefineMeshBlocks` calls `PrepareRegrid` after generating
the new Z-ordered leaf list/load balance and before Step5 modifies native Views.
Planned bounds use exactly the native `LeftEdgeX` calculation and physical-end
branches. The complete cubic dyadic partition, future native count, ng4/N shape,
50000 active-cell limit and1024-cell coarse-solve cap must pass. Metadata uses
the **planned** count/revision, not the old live pack count.

Preparation calls `RebuildCompositeAtAcceptedState`, retaining the old engine,
its adapters, canonical coefficients and native mirror until commit. A new
global hierarchy and frozen geometry are built, then the original full-vector
shift equation and direct boundary data are solved on that geometry. Every
unchanged shell/scri coefficient is checked bitwise. A read-only preflight uses
the exact planned active/ghost mirror arithmetic and physical ADM conversion;
nonfinite/interpolation/ADM failures occur before native field mutation. It
never reads or scatters through the old native pack as if it were the new pack.

The source is the authoritative22 coefficient fields plus a separate three-
component beta warm start. Native25-field unpacked Penrose/Z4 variables are a
verified mirror, not the RK/register/transfer representation. In particular,
subtracting native offsets can lose sub-ulp canonical coefficients, and native
`coarse_u0` is not maintained by the composite task route. The canonical-owner
branch therefore bypasses native Z4c restriction/copy/prolongation during Steps5--9.
The normal Cauchy path and other physics branches remain unchanged.

Commit occurs after Step10 replaces MeshBlock/Coordinates and sets neighbors,
before `InitBoundaryValuesAndPrimitives`. Actual new bounds/levels must match
the prepared map. The owner/pack/Z4c objects persist; the field adapter, core map,
indices, coefficient stage and expected mirror buffers are rebuilt. Old adapters
are released before their old grid owner. The already-certified accepted beta,
all native active/ghost fields, separate physical ADM and new-grid constraints
are installed before the next Driver task can observe the new state.

A data-dependent preflight failure leaves old native Views untouched. The native
tree may already have been updated by the caller; this is a fatal aborted
transaction, not an in-process tree rollback. An impossible planned/actual map
mismatch or late I/O failure after native topology mutation also aborts rather
than continuing from a partial owner. Previous accepted native checkpoints are
not modified. This phase does not claim a general rollback mechanism.

## Explicit accepted RK scratch reset

All coefficient state, accepted time/cycle and actual live Driver RK binding are
preserved. The accepted regrid deliberately resets scratch registers to the
transferred accepted coefficients. This is not an unnoticed history change:
for both native RK3 and RK4, `BeginCycle` resets the register abscissa, and
`BeginStage(1)` replaces every scratch group with current values before a rate
or mixture can consume it. Old accepted scratch values therefore cannot affect
the next cycle. Beta remains separate and is never reset/mixed as a Gamma driver.

`test_native_register_reset.cpp` parses actual Driver coefficients and exercises
a nontrivial coupled, time-forced linear ODE over volume/scri/arbitrary endpoint
groups. All seven next-stage comparisons across RK3/RK4 are bitwise, including
stage1 scratch values and final coefficients. Time/cycle/RK/beta are unchanged;
a midcycle reset rejects. The native regular owner currently has no extra
endpoint fields; wormhole/puncture endpoint transfer is not inferred from this
component proof. Actual regrid sidecars additionally verify geometry epoch
increments **exactly once**, future leaf count/revision binding, exact accepted
scratch reset, unchanged shell/scri coefficients and actual RK arrays.

## Output and restart lifecycle

Ordinary accepted snapshots and their nativeRST are pre-regrid at that cycle.
Explicit `.regrid-1` and `.regrid-2` snapshots/sidecars are post-transaction. The
first independent replay exposed that a repeated `PrepareNative` diagnostic
could overwrite the ordinary cycle CSV with the new topology while nativeRST
retained the old one. An adaptive-only same-cycle diagnostic guard fixes this;
the complete failed run and mismatching4320/5832 shapes remain recorded.

A second replay setup subtracted rounded cumulative JSON times, perturbing third
and fourth dt by one ULP and producing differences near1e-22. Corrected replays
read the exact binary nativeRST `dt` and retain a strict bitwise target. No
production timestep, scientific tolerance or equation was changed for that fix.

Adaptive sidecars use distinct magic and bind complete native leaf topology,
revision, initial gauge-pulse identity, explicit validation schedule and actual
RK/register/global state. The persisted input revision reconstructs matching
caches before restore. Restart invalidates geometry once and re-solves the
accepted shift. All numerical checkpoint data are bitwise after restart except
that required+1 geometry epoch/checksum; complete sidecar-byte identity is not
claimed.

The inherited `pzoom` constructor check preceded actual late zoom creation in
`main.cpp`. A new explicit adaptive `cyclic_zoom` criterion check rejects it
before that object can be constructed. The first guard build's InputBlock
value/pointer compile typo is retained; the corrected frozen-header guard build
and actual negative execution pass. No shared header was read by that private
relink, and no shared build was changed by it.

## Actual Driver controls and evidence

All numerical processes use one local Kokkos Serial CPU and at most180 seconds.
N6/ng4, root2 with one negative octant refined at cycle1 and coarsened at cycle3
exercises Cartesian fine/coarse faces/edges/corners touching the cube/shell
interface. Native count is8->15->8. With two shells global cells are
4320->5832->4320; with one shell3024->4536->3024. Shell maps and all22 shell/scri
coefficients remain bitwise through each transfer; no angular/radial averaging
or symmetry enforcement enters evolution.

The cycle scheduler is an explicit **validation-only** option, disabled by
default. Ordinary criteria remain in place when absent; their general physical
adaptivity is not qualified by this scheduled control. Native max/root/hysteresis
masks remain active. All four controls below perform real native refinement,
coarsening and a restart from the refined native checkpoint at cycle2.

| Case | Full Driver | Restart | Same-engine actual RK/topology replay |
|---|---:|---:|---:|
| RK4/two-shell angular lapse amplitude1e-6 |28.00s|15.27s|22.05s|
| RK3/one-shell Minkowski equilibrium |16.37s|9.17s|11.50s|
| RK3/one-shell CFL-active equilibrium |16.33s|9.19s|11.48s|
| RK4/two-shell angular lapse amplitude1e-3 |41.21s|21.07s|34.30s|

Every replay uses actual native topology and live Driver coefficient receipts,
exact nativeRST timesteps, frozen project/vendor headers and hashed libraries.
Compiler dependency probes report zero live project-header leaks. Each matches
all22 volume/scri coefficients and canonical beta bitwise on21 ordinary/event
comparisons. This checks recurrence/transaction dispatch with the same engine;
it is not an independent PDE derivation.

Independent analysis verifies all25 unpacked native active fields against global
coefficients and fullprecision nativeRST on both refined-cycle2 and
coarsened-cycle4 packs. Independent Penrose-to-ADM conversion includes every
native ghost point. All quantized outputs are bitwise and meet the unchanged
3e-7 rounding target. Final and coarsening-event physical snapshots, all numerical
register/RK/shift/metadata and accepted time are bitwise on refined restart.
Unopted adaptive use, wormhole adaptive use and changed validation schedules
reject at their intended guards.

Qualified directories are `native-adaptive-pulse-qualified-20261027`,
`native-adaptive-rk3-equilibrium-20261027`, `native-adaptive-cfl-20261027` and
`native-adaptive-strong-pulse-20261027`, with separate `*-reference*` directories.
`native-adaptive-pulse-first-20261027`, `native-adaptive-pulse-reference-20261027`
and the earlier directory named `native-adaptive-pulse-reference-qualified-20261027`
retain the output/timestamp failures; their names do not establish qualification.

## Fresh-grid timestep proof

No timestep source fix was needed: after commit and boundary initialization,
`AdaptiveMeshRefinement` calls Z4c `NewTimeStep`; the native owner evaluates the
full new-grid characteristic, SAT and relaxation bounds. Driver then calls
`Mesh::NewTimeStep`, including its ordinary maximum2x growth limit.

With cfl=.005, ceiling=.0005 and one shell, exact used dt values are
`[4.261105759405178e-5, 2.8066292246365876e-5,
2.8066292246365876e-5, 4.261105759405178e-5]`.
They drop on refinement and rise within the fresh coarsened limit. Neither SAT
nor telegraph relaxation bounds were disabled. `cfl-proof.json` compares them
to the post-commit full-grid limits without subtracting rounded timestamps.

## Residuals, general-metric witness and scientific limits

All H/M/Theta/Z diagnostics retain every active cell, with the existing Penrose
proper-volume weighting and kernel normalization. Final RK3 equilibrium H RMS is
4.43e-16 and H max1.89e-15; tensor scri pole1.20e-15. The small angular pulse
has H RMS3.67e-12, M RMS1.37e-10 and tensor pole1.15e-11.

For amplitude1e-3, final H RMS3.6685e-9/H max8.3972e-8, M RMS1.3739e-7/M max
1.4014e-6, Theta RMS7.0638e-13 and Z RMS2.6379e-11. Original vector solve RMS
maximum8.5767e-12 and point maximum6.0828e-10 retain the original targets. The
tensor pole is1.1483e-8; clock/null/tangential residuals remain at roundoff.
These coarse short controls do not establish spatial/temporal/late-front
convergence, constraint accuracy at the first scientific milestone, or collapse.
No tolerance is weakened and no core/interface/scri cell is omitted.

On the bitwise-matched accepted strong-pulse jets, a **diagnostic-only** full
Lg beta minus flat L0 beta comparison has weighted RMS5.2883e-10 and point
max8.2461e-9, above the corresponding solve errors. The metric anisotropy is
1.9154e-11. No flat operator is used to solve or replace the gauge. This is a
pure lapse gauge perturbation, not a transverse-traceless gravitational wave;
its small anisotropy does not replace independent curved-metric manufactured
verification or a future physical nonspherical benchmark.

## Source revision and reproduction scope

The complete four-control/replay phase uses frozen binary5c8916bdd1d40142b06e114133e57b5ed162a92fa0415d710990e616df0d3f47,
with source/config hashes in `native-adaptive-build-preflight-20261027`.
The later shared build against current default-OFF affine headers is recorded
separately in `native-adaptive-final-shared-build-20261027`. Native inputs do not
expose the affine option. No affine-ON native evolution is claimed by these tests.
The parent owns that independent operator work and any future native exposure.

Use fresh output directories, the recorded build capsule and one CPU:

```sh
python3 tst/elliptic_shift_3d/run_native_adaptive.py --output NEW_RK4
python3 tst/elliptic_shift_3d/analyze_native_adaptive.py NEW_RK4
python3 tst/elliptic_shift_3d/qualify_native_adaptive.py NEW_RK4 --output NEW_REFERENCE
python3 tst/elliptic_shift_3d/run_native_adaptive.py --integrator rk3 --shells 1 --amplitude 0 --output NEW_RK3
python3 tst/elliptic_shift_3d/run_native_adaptive.py --integrator rk3 --shells 1 --amplitude 0 --cfl .005 --ceiling .0005 --output NEW_CFL
python3 tst/elliptic_shift_3d/run_native_adaptive.py --amplitude .001 --output NEW_STRONG
```

Frozen executables are local rebuild artifacts and are not shipped in Git.
Source/configuration and exact commands are retained for rebuilding them.
Existing1D jobs/worktrees were untouched; no remote allocations, GPU jobs,
pushes or external PRs were made.

## Final shared/default-OFF qualification

The final shared native target built in8.79s with no source changes during the
build. Its immutable byte-identical capture has SHA256
`a4b4485269f640cf8e478d4416336bfdcae5ca41bac53d1dfd1355adefd66f43`.
The actual small angular adaptive case completes29.55s plus15.43s refined
restart. All17 ordinary/event/coordinate CSVs and all seven complete accepted
sidecars are bitwise against the independently qualified5c8916 phase. Thus the
current default-OFF header path and zoom guard preserve the qualified numerical
state, registers, actual RK arrays, geometry epochs and canonical beta. The
latest input still does not enable affine beta; no affine-ON native claim follows.

The final source capsule freezes actual compiler dependencies separately without
changing hashes of the runtime build receipt. The regular/static default
regression is recorded in `native-adaptive-final-default-regression-20261027`.
All source and failed attempts remain available for review. Executables/objects
are rebuild artifacts and are omitted from the Git-facing evidence package.


The final uniform ng3 and static-SMR ng4 regressions complete2.93s and18.00s,
respectively; all six snapshot CSVs per case remain bitwise against established
controls. Exact refined-cycle2 and coarsened-cycle4 diagnostics separately report
bulk, Cartesian block/coarse-fine seams, Cartesian/shell coupling, radial shell
seams, angular face seams and the scri stencil band. Detailed bands overlap;
`bulk + any_interface_stencil_band` is a complete disjoint partition. Empty
coarse-fine regions after coarsening are explicitly counted, not omitted from
full-domain acceptance. The final native build source capture has187 actual
compiler dependency files and681 source/vendor/generated files, with370
applicable recorded pre/post source hashes verified.

## Lossless evidence package

The [native adaptive capsule](evidence/native-adaptive-package-20261027/README.md)
ships all 22 phase directories, including failures, without executables or
private build trees. Scientific data are deduplicated losslessly into eight
archive fragments, each at most 32 MiB. Independent restoration from the fragments
verified all 564 scientific files and 2,999 archived source members by SHA256.
The raw scientific data total 1,130,355,514 bytes; the compressed chunk archive
is 241,099,985 bytes. All local originals remain intact. Source and evidence
paths/hashes are listed separately in the capsule's `ready-for-commit.json`.

This is qualification of the restricted regular native lifecycle. It does not
complete the requested wormhole, moving puncture, MPI/device or full scientific
convergence milestones. Those restrictions remain explicit in the native owner.

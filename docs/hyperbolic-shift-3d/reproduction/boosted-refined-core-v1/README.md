# Boosted hole: uniform Cartesian core refinement

## Current correction: matched Cartesian CFL (BC2)

The user requested matching the original Cartesian CFL. Use dt=.00025 with
the doubled core resolution. BC1 job9036322 was cancelled during allocation
startup (PBS F/143, no worker/native output); its matching root lock is released.
Do not resubmit BC1. Preserve its frozen records as a superseded experiment.

BC2 keeps the original effective scri penalty coefficient explicitly fixed at
.5*(1-1e-12)/.768=.6510416666660156. Thus p4/p8 and the boundary equation match
BG1/BC1, while the actual boundary penalty CFL becomes.25 (unlimited nominal.384).
Simply halving dt without this override would restore full-strength penalties
and change the boundary equation. BC2 has distinct checkpoint identity and
source/build/launcher-boosted-refined-core-cfl-v1. All other science and the
2node/24rank allocation remain unchanged. No failed-state restart or GPU smoke.
Local BC2 two-step check passed: CFL<.16, finite admitted checkpoint, null<1.3e-15.
Current source cb0f47b520afe4f88c98e11569f3ba88e632df9a,
manifestSHA2566d1c29a69bcce98d761d1f896b9f230553d7e4852d8a4c86384f33237634847b.
Launcher fef313e5a; inspect state.json for full build/job status.

Current BC2 job **9036411** is submitted using frozen launcher
`launcher-boosted-refined-core-cfl-v2` (commit187b02e037ab9df8d45725858e0509e3b29c3acf).
Two debug-scaling nodes /24 GPU tiles, fresh t0 to10M, dt=.00025. Plan
`campaigns/n16-boosted-refined-core-cfl-v2-c16-evolution-segment01/plan.json`,
SHA256 e42860fa9db75d196e1677db78babfed901aa29d4417abb372456d60e7949537.
Both queues were empty before retry; footprint6.789GB, headroom2.517GB after
output/atomic reserves under10GB. Source/build/binaries/science are unchanged;
v2 intentionally binds the original v1 science file, SHA256
b0124b10763a01abc9e1feb0436b6022078f320ad6c53ec40e989231cfdab17c.
Actual evolution binary SHA256
fbb8f8d2e9522d1b72fe28ea8b1e857f2b840f4d58365c5042a16727831f59e6.
Use v2 controller/helper for current/future plans; do not rebuild or refreeze.

Prior BC2 job9036342 failed before launching any native evolution. Its affinity
probe produced correct two-node placement,12 local ranks/unique GPU masks each.
The old validator incorrectly demanded ranks0-23 on a single host. PBS F/1,
worker cases=[] and only affinity.log establish an infrastructure failure,
not a numerical instability. Frozen old records are preserved under
`affinity-failure/`; matching terminal lock released. The v2 validator checks
local ranks0-11 and masks on each allocated host. Three focused regression tests
passed, including the actual saved Aurora probe and negative malformed-placement
cases. No scientific code or parameters changed. Fresh job9036411 replaces it;
never resubmit9036342 or mutate its frozen launcher. See cfl-v2-* records and
launcher-v2-provenance.json. No native checkpoint existed to resume.

Hourly monitor aurora-boosted-refined-core-to-10m is ACTIVE for BC2 only.
Use the frozen cfl launcher/controller/helper for exact BC2 configuration,
finite payload, source/binary/input/artifact hash, dt/time/steps/progress,
tracker and fixed-penalty verification plus scheduler/worker/native consistency.
Healthy admitted walltime checkpoints below10M continue sequentially with the
same C16 identity, verified checkpoint/hash and horizon seed, after matching
terminal lock release; inspect both user queues and saved submissions first.
Numerical failure is terminal and must be preserved/localized, never replayed.
Keep current restart until successor verification; no retirement without local
hash/length preservation and fresh remote rehash receipt. No parameter changes.
Compare to BG1 through4.23M, preserving horizon gaps, trajectory, constraints,
null/pole/clock and segment-summed floor/sign counts with boundary counts separate.
Use matching diagnostics only on admitted endpoints. BC2 refines space and time
together at fixed effective boundary penalty; survival alone proves neither
convergence nor purely spatial attribution. Inspect figures, commit/push evidence,
then remove monitor after full terminal follow-up. No AMR implementation here;
AMR-HANDOFF.md is the separate-thread/new-worktree prompt requested by the user.

## Historical BC1 preparation (superseded)




Branch `codex/boosted-refined-core`. User authorized uniformly refining the
Cartesian core and a fresh evolution, followed by work on full Cartesian AMR
reusing AthenaK infrastructure. This fixed-grid experiment must be submitted
before AMR development; it is not itself a dynamic AMR test.

BC1 preserves BG1's exact v=+.1 boosted geometry, free initial gauge, six outer
shells at N16, broad restoring collar .6-.9, C6 initial slice4M-12M, inner gauge,
KO8, original scri operators, floors, tracker and moving-center fast-flow.
Only the Cartesian core is spatially refined: 4x4x4 blocks of16^3 cells instead
of2x2x2. The core has64^3 nominal cells, compact spacing .005625 versus .01125,
physical initial lab spacing .0551214M. The cube and shell overlap radii are
unchanged. The global Cartesian stencil policy is retained, with no internal
coarse-fine interface. Surface grid and all six shell maps are unchanged.
A distinct BC1 208-byte checkpoint binds core subdivisions4; BG1 is rejected.
Never restart the failed BG1 state. Fresh native initialization only.

Planned dt=.0005 retains the coarse run's timestep and scri penalty scale
.651041666666; initial/two-step CFL must be measured before submission. This
is a spatial test, not a reduced-timestep test. Target10M (coordinate10/3),
M1/3, diagnostics.05, horizon12/24/tol.001, angular2/scri nominal1. No GPU smoke
gate per existing user preference. Local mesh/finite-payload/coupling checks
are required for the newly changed layout.

User additionally requested more debug-scaling nodes: use2nodes with12 ranks
and GPU tiles per node (24total), with explicit allocation/node/rank checks.
Keep unrelated held jobs. Aurora account CompactBinaryMerger, root
/home/hzhu/hyperbolic-core-shell-20261028, python/usr/bin/python3.11,
make-j64, immutable source/build/launcher, authorized10GB including reserves.
No source/build/launcher or unverified output deletion. Healthy verified BC1
walltime checkpoints may continue sequentially with horizon seed. Numerical
failure is terminal, preserve/localize and never replay. No other resolution
or gauge variant is automatically authorized by this isolated experiment.

Compare to BG1 through its last accepted4.23M, preserving its failure and
moving-horizon gaps. Success beyond4.23M would show sensitivity to core spatial
resolution at fixed dt/boundary penalty, not prove convergence or isolate all
gauge effects. Save trajectory, mass/acceptance, constraints, null/pole/clock,
shift/metric behavior and sign/floor counts. Only admitted endpoints may enter
common-cutoff reconstruction. Preserve scientific snapshots separately from
subsequent AMR implementation, and maintain the hourly quiet continuation monitor.

Local pre-submission checks passed with corrected geometry-based tracker ownership:
521056 evolved cells,100 blocks,259205840-byte BC1 checkpoint. One- and two-rank
CPU SERIAL two-step checkpoints are byte-identical (SHA256
f9619aa9baa194a2cec405b60ec26c65b6f68058a62aa668259cb56d001a2ff5).
Maximum CFL .3080336425, null surface/ghost<=1.111e-15, accepted horizon mass
.333333343511. The expected .651041666666 penalty scale is unchanged. These
are local layout/physics checks, not a new GPU smoke or a long stability result.
The initial local attempt exposed a tracker guard hardcoded to eight core blocks;
this was corrected to inspect Cartesian block geometry before source freezing.
The production source is022c2adbfd5d94be3b2a35755bb4a93462b118d8,
launcher71ac666bab9c17457f847d9f6bf916db903f332f. Earlier local failed artifacts
are retained under output/boosted-refined-core/serial-one; they were never submitted.

Fresh BC1 C16 job **9036322** submitted on2debug-scalingnodes/24GPUtiles.
Plan `campaigns/n16-boosted-refined-core-v1-c16-evolution-segment01/plan.json`,
SHA2560200064ec123528d932ba1cd40b19e6e574c91e39c6587755a5f5660aed7a7ee.
Frozen scienceSHA2564338cd1dcc6378835e773774c2d4ddddf51a1d5719a931cba9f6aa604d8d0e62.
All launcher and build provenance locally preserved and verified in aurora-build/.
Both user debugqueues were empty before this single submission. Bindingfootprint
6.697GB with2.609GB headroom after reserves; no purge. Hourly monitor
`aurora-boosted-refined-core-to-10m` is active. Scientific snapshots remain
immutable while subsequent Cartesian AMR integration uses a separate branch.

User subsequently requested AMR implementation in another thread/new worktree.
No AMR implementation is to be done in this campaign thread; use AMR-HANDOFF.md.

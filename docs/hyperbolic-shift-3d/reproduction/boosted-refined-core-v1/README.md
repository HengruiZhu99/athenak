# Boosted hole: uniform Cartesian core refinement

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

# Larger blocks and timestep performance control

The user prioritizes24³ meshblocks with two interior refinement levels, merging
shells to approximately preserve radial resolution. This is the next bounded
GPU experiment, before the requested stable roughly4× GPU-hour gate and the
boosted10M resolution family. No speedup is established yet.

The original12³/three-level mesh has400 Cartesian leaves and432 shell blocks.
The native AthenaK MPI2 topology check for24³/two-level has176 Cartesian leaves
and78 shell blocks (13 radial layers),254 total. Finest spacing is identical:
`0.8/(4*12*2^3) = 0.8/(4*24*2^2) = 1/480`. The origin's finest support radius
increases from.05 to.10. Larger blocks necessarily refine more volume: evolved
cells rise from1,513,728 to3,750,624 (2.47774×). It is not a pure repartition.
The coarse Cartesian spacing halves. The figure `large-block-control/layout.png`
was inspected; it shows block boundaries, not individual cell lines.

Pairs in .78–.95 merge with exactly preserved Δr/N. Their old angular factor2
atN12 becomes factor1 atN24, retaining the same angular points. Other interior
pairs approximately preserve radial spacing while retaining the Omega/gauge
transition endpoints. The first .35–.40 and last .95–1 shells remain separate;
their radial spacing halves. Inner and scri angular resolution also doubles.
This changes the scri penalty rate through its mesh spacing; its multiplier
and equations remain frozen. Keep that distinction in any timestep comparison.

The proposed same-allocation four-node pair uses the *same frozen executable*,
fresh exact v=.1 data, dt1/24000 and target coordinate.025=.075M (600 steps).
Only mesh/configuration differ. Compare stepping timers separately from setup,
checkpoint I/O and allocation cost, plus fixed-r>.3 constraints, mass/acceptance,
tracker, floors, scri residuals and common-coordinate fields. A600-step result
cannot establish10M stability or convergence. No checkpoint from this benchmark
is automatically reused in a long physics run. Both cases require unchanged
frozen native verification; numerical failures stop and are not replayed.

## CFL audit

The confirmed Cartesian reference is `origin/project/telegrapher_lapse`,
commit b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a. Its `z4c_newdt.cpp` takes the
minimum directional dx/speed and a telegrapher relaxation bound; the mesh then
applies cfl_number. This is not a stability certificate for our mapped shells.
Our native estimate sums mapped directional advection plus characteristic speeds.

Read-only reconstruction on the admitted two-step boosted12³ state identifies
shell layer18 near r=.921615 as the limiter. CFL=.1937154374 at dt1/24000;
radial contribution=.1918881711, each angular contribution=.0009136332.
A Euclidean norm would change the timestep by only1.00950×. Thin radial spacing,
not the Cartesian core or three equal directional terms, dominates this state.
The reconstructed final-point value differs from the recorded cumulative stage
maximum by2.84e-9; they are different sampling times. No production code changed.

The independent scalar frozen radial advection/wave RK4 screen passes4× and5×dt,
with a sampled stability edge near6.785× for that scalar model. It excludes full
Z4c coupling, non-normality, background gradients, KO, sources, interfaces, SMR
and scri; it only motivates bounded evolution tests. An opt-in L1-CFL-limit
proposal is saved as an **unapplied, untested patch** in `cfl-audit/`. Existing
source, guards and frozen binaries retain their original limits. Test larger
blocks first; do not silently replace L1 with L2 or claim4×GPU savings from dt.

The bulk device kernel already batches owned active cells across blocks; this
is not one GPU launch per small block. Larger blocks principally change halo/
interface work, mesh occupancy and volume, so profile those costs explicitly.
Five obsolete remote output copies were freshly hash/length verified locally
and rehashed remotely before retirement (2,864,555,616 logical bytes); receipt
retired-before-large-block-pair.json preserves all identities. Local originals,
scientific snapshots and current restart inputs remain intact.

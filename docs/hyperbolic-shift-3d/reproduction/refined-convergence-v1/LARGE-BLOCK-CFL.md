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

## Submission and retained stationary state

Four-node48-rank job9040972 is submitted (last schedulerQ), plan SHA256
10e7b338f6f32954b38af7d2a7b69700e6d0a996d1eaf031f6ca06bc268d7239.
Both debug queues were empty before its single submission. Binding footprint
40,501,058,186bytes; headroom3,548,941,814bytes after planned/atomic reserves.
Monitor this existing job; never duplicate it or restart a failed benchmark.

Stationary9040850 ended healthy at4.027125M/32217steps. Exact frozen verification
and all152 local artifact hashes/lengths pass; matching terminal lock released.
The retained413,302,286byte checkpoint has SHA256
e240a748dfd043ad905c8681c3282d1fb8bbe334345af7ac18f0c1d2d54313d3.
Do not automatically continue it ahead of the requested performance controls.

The user additionally asks to reduce the need for dense outer radial sampling.
Next initial-data controls should jointly redesign the slicing transition and
radial map, regenerate the exact geometric data and compatible reference gauge,
and test derivatives, spacelike/map margins, constraints and unchanged scri
limits before evolutions. The measured lapse-gradient half-width is.0213 in
compact radius, distinct from the Omega flattening join. A coordinate stretch
alone also transforms characteristic speeds and is not a guaranteed CFL gain.
A stationary-reference subtraction in the discrete RHS is a separate possible
control; it must retain perturbation dynamics and full-field diagnostics.
The submitted block comparison stays unchanged and does not establish that
coarsening the dense band is safe.

The first analysis-only joint slicing/map control is complete; see
COMPACT-SLICING-STUDY.md. At fixed central20M boundary and isotropic scale,
compact-coordinate C6 switches ending at.95/.97 reduce the peak lapse gradient
29.48→18.30/13.53. Independent radial constraint-differencing errors decrease
substantially, with native baseline fields cross-checked to7.51e-13 scaled.
This is not a production/boosted evolution, timestep or performance result.
The broader physical slicing ends at58.46M/98.48M respectively; retain that
change explicitly. The frozen block benchmark remains unchanged.

## Completed benchmark and next attribution

Job9040972 is terminal and fully preserved: all303 artifacts verified.
The24³ layout improves per-cell throughput1.386× but evolves2.478× more cells;
native command time is1.834× larger. Both short evolutions show sharp exterior
constraint growth concentrated near r=.95. See TAIL-INTERFACE-ATTRIBUTION.md
for native volume localization and the existing fresh four-node control
job9041071. Do not resubmit9040972 or treat either benchmark as10M stability.
LAPSE-REGULARITY-OPTIONS.md separates normalization from physical lapse collapse
and records consistent equation/discretization candidates.

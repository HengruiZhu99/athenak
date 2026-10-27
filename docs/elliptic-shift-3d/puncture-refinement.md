# Geometry-only tracked-puncture Cartesian refinement

`RefinePunctureNeighborhood` accepts a `CompositeMesh` **by value**, a
`PuncturePosition`, an absolute target refinement level, and an active-cell budget
(default 50000). It returns `PunctureRefinementResult`, whose `.mesh` is the new
descriptor. Counts include active cells, Cartesian/shell leaves, targeted and
balancing refinements, containing leaves, minimum Cartesian spacing, old/new
revision and a position stamp. No fields or solver objects are inputs.

The helper refines every closed Cartesian leaf containing the tracked position
until its level is at least the target. An origin at the eight initial block
corners consequently selects all eight, without choosing one privileged octant.
Ties use exact inherited map bounds. There is no tolerance snap: a nonzero
`nextafter(0,1)` coordinate belongs on its actual side. The point, its velocity,
time, stage and revision are unchanged. No singular power, regular-center parity
or shift condition is introduced.

After target refinement, all Cartesian pairs with closed intervals meeting at
a face, edge or corner are checked. Whenever levels differ by more than one,
the coarser leaf is refined. Each pass sorts and deduplicates the coarse indices
and applies them in reverse order; this keeps indices valid and makes the result
deterministic. Balancing can refine Cartesian leaves beyond the target path,
including neighbors away from the point. It never coarsens an existing leaf or
refines a shell. A target below existing refinement is a minimum request; global
balancing still repairs any Cartesian contact imbalance.

This operates on the octree levels and exact inherited bounds of the existing
`CompositeMesh` Cartesian descriptor. Every leaf retains the same logical
N×N×N shape and ghost contract. Global half-width, radii, shell maps, Omega and
reference/gauge conventions are untouched. The 2:1 rule here covers Cartesian
leaves; it does not introduce shell AMR or prove Cartesian/shell transfer
accuracy.

Targets outside [0,30], invalid puncture kinematics, a point outside all Cartesian
leaves, invalid Cartesian map bounds/levels, overlapping Cartesian interiors,
unrepresentable midpoints, revision exhaustion, cell-count overflow and budget
exhaustion reject explicitly. An initial mesh already exceeding the budget also
rejects. Every failure affects only the local copy; the caller's original mesh
is unchanged even if several refinements preceded failure. The budget includes
**all active cells**, including shells, and excludes ghost allocation. It is a
cell-count guard rather than a complete memory/runtime estimator. All counters
and the revision reflect successful parent-to-eight-child operations.

The owner must transfer/prolong/restrict fields and rebuild mesh, halo,
multigrid, puncture derivative and evolution-stage state after adopting the new
descriptor. None of those steps is implemented here. In particular, this helper
does not evolve the tracker, translate the global lapse source, impose a puncture
endpoint value, or establish safe moving-puncture evolution.

## Bounded verification

Run after the existing CPU build is available:

    python3 docs/elliptic-shift-3d/evidence/puncture-refinement-20261027/reproduce.py

The standalone test uses the actual descriptor with N=6, 2³ initial Cartesian
blocks, and seven shell radii `.45,.55,.60,.65,.70,.75,1`. All 42 shell descriptors
remain bitwise unchanged. Results at the fixed origin are:

| target level | Cartesian leaves | all active cells | minimum Cartesian spacing | revision increment |
|---|---:|---:|---:|---:|
| 0 | 8 | 10800 | .0416666667 | 0 |
| 1 | 64 | 22896 | .0208333333 | 8 |
| 2 | 120 | 34992 | .0104166667 | 16 |
| 3 | 176 | 47088 | .0052083333 | 24 |

All eight containing leaves reach the requested level. These origin cases are
already balanced after targeted refinement. The independently checked final
face/edge/corner contact counts at level3 are 552/768/316, with every level
difference at most one.

For X=(.037,.013,−.021), target3 uses 3 targeted plus 14 balancing refinements,
36504 total cells, one containing leaf and the same minimum spacing. Its final
378 face, 516 edge and 216 corner contacts are all balanced. This is a geometry
translation check, not a translated gauge or black-hole solution.

Every returned active cell center is mapped through the original descriptor's
inverse and forward map; the maximum physical coordinate error is 2.23e−16.
The Cartesian partition volume and its integral of the constant field are
unchanged (exactly zero reported defect). This checks geometry/partition
preservation; **it does not test transferring a stored constant field**. The
tracked physical position also roundtrips unchanged.

Repeated calls from the same input are deterministic. Applying the helper again
to its balanced result at the same target changes no descriptor or revision.
Invalid targets, insufficient budgets (including 47087 for the 47088-cell
request), shell/outside-domain points and nonfinite positions reject without
changing the original. The exact 47088 budget succeeds. A face-tie query selects
two leaves while its nextafter offset selects one, demonstrating that point
coordinates are not snapped or floored.

Full results, stderr, source hashes, source HEAD, commands and timing are retained
in the evidence directory. No coupled evolution, field transfer, coefficient
coarsening, multigrid convergence, MPI/GPU, restart or radial-collapse acceptance
claim follows from these geometry tests. The intended next use is a separately
validated lower-resolution radial pilot with more Cartesian core resolution;
core physical residuals and all original scientific gates remain required.

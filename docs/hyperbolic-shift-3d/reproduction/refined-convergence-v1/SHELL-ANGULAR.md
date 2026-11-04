# Localized shell angular refinement and boosted initial resolution

This is a separate local candidate on `codex/refined-convergence`; it does not
replace the frozen stationary Aurora source, binary, plans or existing job.
The full goal remains regular, stable, convergent stationary and exact v=.1
boosted punctures through10M. The controls here cover only analytic initial
reconstruction and two local Serial-MPI steps (0.00025M).

## Mesh and interface implementation

Optional `shell_angular_subdivisions` supplies one factor per radial layer.
Each factor is1 or2. Factor2 splits each of the six cubed-sphere faces into
four blocks, doubling angular sampling at unchanged radial sampling and local
block size. Cartesian geometry/refinement is unchanged. Factors must be1
inside the entire evolved Cartesian corner radius and in the outermost shell.
Thus the existing scri surface, normal closure and angular dissipation remain
on precisely their previous grid. Missing input preserves the old identity
and mesh; explicit factors are bound into the RF2 formulation identity.

`CompositeMesh::SubdivideShellAngles` reuses the native forward/inverse maps,
Jacobians/Hessians, block bounds and physical donor ownership. The native
AthenaK Cartesian AMR adapter retains the factors when reconstructing a
descriptor, including after restore. No new time interpolation, subcycling,
field basis, derivative order, gauge equation, conformal factor or floor is
introduced. Dynamic regridding with these shells is not qualified here; the
production candidate uses the separately qualified fixed SMR tube.

The previous radial interpolation chain selected only layers with identical
angular bounds. That is correct for the old uniform angular layout, but skips
neighboring layers when a newly subdivided face meets an unsubdivided one.
For mixed layouts, the adapter now selects the nearest six physical radial
planes, then performs six-point interpolation in each donor plane's own
angular tile. The value is

\[
 u(r,q^1,q^2)=\sum_{k=0}^{5}L_k(r)
       \sum_{j,l=0}^{5} A^{(k)}_j(q^1)B^{(k)}_l(q^2)u_{kjl}.
\]

All donors are active values in the same Cartesian component basis. Existing
sparse MPI request plans consume these entries; no new communication protocol
is used. Uniform angular layouts retain the original numerical path exactly.
This fixes an assumption exposed by the new mixed angular layout; it is not
claimed to explain failures of the earlier uniform-layout campaigns.

## Qualification and provenance

Separate candidate attempt02 binary:
`66216ed21f8fa28a42425ed27620a323a8290baf797a1aa33bfd8a4d538f9cd1`.
Build commands and source/header/archive hashes are in `shell-angular-checks/`.
The frozen local production executable hash remains unchanged.

- MPI1/2 maps, ownership and AMR descriptors agree. Map/Jacobian/Hessian errors
  relative to the original chart are at roundoff. Nearest-plane donor tests
  cross both coarse/fine radial interfaces and reproduce a separable degree5
  radial/angular polynomial to1.12e-16. They explicitly require donors from
  both neighboring layers.
- Malformed factors, refinement in the Cartesian region, changed outer scri
  resolution, and a restart with the wrong angular identity reject.
- The old uniform-layout two-step checkpoint is byte-identical with the new
  executable. All eight saved state arrays are byte-identical between corrected
  MPI1 uninterrupted and corrected MPI2 checkpoint/restart at step2.
- The first angular MPI1 case launched with attempt01 before the donor fix was
  built, despite an attempted coordinator stop. Timestamp and binary evidence
  are preserved in `donor-fix-disposition.json`; its comparison against the new
  binary correctly failed. It is retained solely as an uncorrected-donor
  control, not matching-binary parity. The corrected MPI1 case was run under a
  fresh prefix; no data were overwritten or failure restarted. The actual
  corrected verification is `shell-angular-checks/verification.json`.

The physics controls deliberately use the same uniform coarse N12 Cartesian
core to isolate the exterior change with bounded local memory. Their horizon
finder is not successful at its online tolerance. They are not an acceptable
production puncture grid. The eventual candidate retains the400-leaf symmetric
SMR tube and must be verified as a combined configuration before long evolution.
No GPU check, new Aurora build or new Aurora submission was performed.

## Measured initial error attribution

For radial18 with angular sampling doubled only in .78<r<.95, the fixed
r>.3 initial H/M norms are.00310066607/.199621473. Native M is the conformal-
metric contraction, distinct from the physical-M norm in the earlier independent
fixed-point controls. CPU reconstruction reproduces native volume integrals.

80.1315% of exterior H² and99.6375% of M² lie in .78<r<.90. Replacing1,561,952
outer ghost values with exact analytic initial values at .70<r<.975 changes
H by−.04087% and M by+.005848%. Active data, operators and scri are unchanged.
This supports dominant bulk truncation in this initial control, not dominant
halo interpolation. It is not a statement about evolved errors or all seams.
See `shell-angular-initial-diagnostic/`; empty radial bins remain gaps in plots.

The next radial24 control divides each of .78–.875 and .875–.90 into four
radial layers. Angular refinement remains2 in .78–.95. Source, initial data,
core, dt=1/24000, equations and outermost shell are unchanged. Native two-step
finite/checkpoint/config checks pass; maximum CFL is.1937154, with no automatic
timestep change. It is a fresh initialization, not a restart from the other mesh.

| Mesh at N12 | Initial H, r>.3 | Initial M, r>.3 | H after2 steps | M after2 steps |
|---|---:|---:|---:|---:|
| radial18, original angles | .014295263 | .216511216 | .013996628 | .213178017 |
| radial18, doubled transition angles | .003100666 | .199621473 | .003089411 | .197173463 |
| radial24, doubled transition angles | .001396980 | .012125204 | .001106979 | .011909669 |

The combined targeted change reduces initial H/M by10.233x/17.856x and
step2 H/M by12.644x/17.900x. The additional radial change alone reduces the
remaining initial M by16.463x. These anisotropic controls do not define a single
global grid spacing or establish fourth-order field self-convergence, long-time
stability, GPU equivalence or uniform slicing regularity. Initial geometric
regularity evidence remains separately preserved.

With the full400-leaf SMR tube, the original radial18 grid has508 blocks,
radial18/angular2 has688, and radial24/angular2 has832. These are nominal mesh
counts, not measured run-time speedups or memory guarantees. Refining only the
measured transition layers avoids automatically refining the whole core and
scri. Actual Aurora cost and stability remain to be measured.

## Reproduction and next action

- `build_shell_angular_candidate.py` builds separate corrected attempt02 files.
- `check_shell_angular_candidate.py` reproduces matching-binary geometry,
  rejection, MPI/restart and initial comparisons in fresh output directories.
  `complete_shell_angular_checks.py` preserves the actual incremental completion
  after the recorded attempt01/02 distinction.
- `run_shell_angular_initial_diagnostic.py` and its separate
  `angular_initial_diagnostic.inc` reconstruct initial data read-only on the
  verified checkpoint topology and apply the explicitly labeled halo control.
- `run_boosted_radial24_control.py` performs the fresh two-step radial control;
  `analyze_shell_angular_initial.py` and `analyze_boosted_mesh_controls.py`
  reproduce the inspected figures.

The candidate input is
`central20_boosted_smr_tube_radial24_angular2_reference.cfg`. Do not silently
substitute it for the active stationary radial18 family. Once SSH is renewed,
first reconcile existing stationary job9038758 and its saved submission; only
healthy, fully verified same-case walltime checkpoints may continue. Combined
boosted-SMR qualification, stationary/boosted long runs and resolution studies
remain open. Git push authentication is also outstanding; local commits retain
the work meanwhile.

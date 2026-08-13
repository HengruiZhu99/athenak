# Half-plane SO(2) Cartoon Z4c design

Status: implemented, locally validated qualification candidate; physical GPU
qualification is pending. The frozen baseline
`5760da5893b256d32da5fb13f92ed8e0c102ed39` uses the signed-rho geometry
described in `docs/axisymmetric_cartoon_z4c.md`. The current worktree implements
the replacement defined here, but must not be called production-qualified until
the fresh three-grid Kerr campaign and CPU/CUDA comparison pass.

## Geometry and component convention

The SO(2) evolution domain is the cell-centred half-plane

```text
  parity ghost storage             active storage
 ... -5h/2 -3h/2 -h/2 | h/2 3h/2 5h/2 ...
                       rho=0
```

The root mesh has `x1min=0`, `x1max>0`, and no active negative-rho cell.
`rho=0` is the inner x1 face, never a cell centre.  The negative-rho cells are
derived ghost values only.  The active directions are x1=rho and x2=z; x3 is
collapsed and represents the suppressed Cartesian y direction.

Physical component order is `(rho,z,y)=(0,1,2)`.  The legacy packed names map
as follows:

| packed component | physical component |
| --- | --- |
| X | rho |
| Y | z |
| Z | suppressed y |
| XX | rho-rho |
| XY | rho-z |
| XZ | rho-y |
| YY | z-z |
| YZ | z-y |
| ZZ | y-y |

This mapping follows the existing `CartoonDirection` definition and the
Killing identities in `cartoon_derivatives.hpp`; it is not the usual visual
ordering of an `(x,y,z)` plot.

The half-plane map is a new restart-incompatible coordinate-map schema.  A
signed-plane restart cannot be loaded as half-plane storage because its mesh
topology, integration measure, and active-cell ownership differ.

## Parity and regularity metadata

Reflection across the axis has basis parities

```text
p(rho)=-1, p(z)=+1, p(y)=-1.
```

Scalars are even.  A Cartesian vector component has its basis parity.  An
all-lower or all-upper symmetric rank-2 tensor has the product of its two basis
parities.  In packed order this gives

```text
scalar: +
vector: [-,+,-]
tensor: [XX:+, XY:-, XZ:+, YY:+, YZ:-, ZZ:+].
```

The Z4c state therefore has the following parity table:

| state | parity |
| --- | --- |
| chi, Khat, Theta, alpha | even |
| gXX, gXZ, gYY, gZZ | even |
| gXY, gYZ | odd |
| AXX, AXZ, AYY, AZZ | even |
| AXY, AYZ | odd |
| GammaX, GammaZ | odd |
| GammaY | even |
| betaX, betaZ | odd |
| betaY | even |
| BX, BZ | odd |
| BY | even |

The same tensor table applies to the ADM spatial metric and extrinsic
curvature.  Covariant and contravariant Cartesian vectors have the same parity
under this orthogonal reflection.  Mixed-index tensors require an explicit
trait and must not use the symmetric-tensor helper by assumption.

Parity alone is insufficient for cancellation-sensitive SO(2) expressions.
The derivative provider also distinguishes these regularity classes:

* `EvenScalar`: `F(rho^2,z)`;
* `OddLinear`: `rho F(rho^2,z)`;
* `EvenQuadraticZero`: `rho^2 F(rho^2,z)`;
* `TensorPlanarPair`: `T_rhorho=P+rho^2 Q`, `T_yy=P`;
* `TensorSwirlPair`: `T_rhoy=rho^2 R`, with the rho-z and z-y
  components odd-linear.

## Axis boundary

The axis is an explicit Z4c geometry boundary, not a generic reflecting
boundary.  For active index `is` and ghost depth `g`,

```text
u(is-g-1) = parity(u) * u(is+g).
```

This operation fills all stored z indices (including z ghosts and corners) and
the sole collapsed x3 plane.  It is allocation-free and Kokkos-compatible.
Axis ghosts are regenerated from active cells; they are not sent, received,
restricted, prolonged, initialized independently, or evolved by RK updates.

The current generic `reflect` implementation in
`src/bvals/physics/z4c_bcs.cpp` is not the axis policy: its component list is
for reflection of the legacy coordinate x1 and, for example, treats XZ as odd
while an axis reflection in the `(rho,z,y)` convention makes XZ even.  The new
axis boundary has a separate flag and a parity-trait-driven fill.

Required fill points are:

1. after active initial data are generated, before ADM-to-Z4c derivatives;
2. after each RK stage has updated active cells and after exchange/prolongation,
   before the next `CalcRHS`;
3. after regridding/load balancing/prolongation before any derivative consumer;
4. before constraint, Weyl, curvature, and axis-local diagnostic kernels when
   they are not already ordered after a current fill.

The task graph must name the fill as a dependency immediately before `CalcRHS`;
relying on a fill left over from the previous stage is not the contract.

## Active-coordinate derivatives

Ordinary rho and z first, second, mixed, advective, and dissipation operators
use the existing centred `Dx`, `Dxx`, `Dxy`, `Lx`, and `Diss` machinery.  Near
the axis, their negative-rho samples come only from parity ghosts.  No
polynomial fit or fitted/raw switch is permitted for these derivatives.

For a smooth parity-extended field, the half-plane result must match the same
centred stencil evaluated on the corresponding virtual full plane.  O2/O4/O6
tests cover every half-cell layer touched by the stencil.

## Suppressed-direction derivatives

The continuum SO(2) identities remain those generated by the Killing vector
`-y d_rho + rho d_y`.  For example,

```text
d_y^2 f = d_rho f / rho
d_y V^rho = -V^y/rho
d_y V^y = V^rho/rho
d_y^2 V^rho = d_rho V^rho/rho - V^rho/rho^2.
```

Because active centres have `rho>=h/2`, the bulk quotients never divide by
zero. Every active half-cell uses the same expression assembled from the
ordinary centered radial derivative through exact parity ghosts and the local
field values. There is no special `s=rho^2` reconstruction, fit width, radial
blending, or layer switch. Exact-axis analytic limits remain diagnostic-only:
the production half-plane samples and evolves no active `rho=0` point.

The initial required operator inventory is the set reached by the shared
provider API:

| consumer | required provider operations |
| --- | --- |
| `z4c_calcrhs.cpp` main RHS | scalar first/second/advection; vector first/second/advection; vector divergence; tensor first/second/advection; component dissipation |
| `z4c_adm.cpp` | scalar and tensor first/second derivatives for ADM conversion and constraints |
| `z4c_Sbc.cpp` | scalar/vector first derivatives in the Sommerfeld RHS |
| `z4c_calculate_weyl_scalars.cpp` | scalar/vector/tensor first and second derivatives |
| `curvature_diagnostics.cpp` and derived/history consumers | ADM tensor derivatives through the shared curvature provider |
| `cartoon_meridional_sampler.hpp` | curvature derivatives at axis-adjacent support cells |
| `cartoon_m0_fastflow.cpp` | ADM derivatives sampled by the m=0 horizon adapter |

Every provider branch is recorded in the generated
[`z4c_cartoon_half_plane_operator_table.md`](z4c_cartoon_half_plane_operator_table.md)
table. It contains tensor character, parity/regularity class, centered
primitive, and the implemented bulk quotient. The generator fails if a former
layer-dependent closure helper returns or if the production source and table
change independently.

## Current signed-plane assumptions and migration

The baseline dependencies are:

| area | current signed-plane assumption | half-plane action |
| --- | --- | --- |
| `z4c_symmetry.*` | schema 1 map `signed_rho_z_suppressed_y_v1`; symmetric finite x1 domain; even nx1 and even x1 root blocks | introduce schema-2 half-rho map; require x1min=0, x1max>0, inner-x1 axis boundary, and no signed restart |
| `cartoon_derivatives.hpp` | evaluates signed rho and uses independent side-local fits in the innermost NGHOST layers | use positive rho, parity-centred active derivatives, and direct regularity functionals; remove production fit path after qualification |
| `z4c_bcs.cpp` | no physical axis; generic reflect uses an ad-hoc component list | add explicit parity-driven axis fill for Z4c and ADM carriers |
| `z4c_tasks.cpp` and driver initialization | physical BCs are applied after receive and before prolongation; no explicit pre-RHS parity invariant | add an axis-fill task/dependency and initialization/regrid fills |
| `mesh_refinement.cpp` | mirrors every refinement flag and requires a symmetric signed-rho tree | delete mirror reconciliation for half-plane; refine only physical leaves; parity-fill axis halos after AMR operations |
| boundary exchange and AMR | negative side is an ordinary active neighbor/tree | inner x1 is a physical axis with no MPI neighbor; restrict/prolong active half-plane data only |
| `kerr_puncture.cpp` | fills all signed active cells and every stored ghost analytically | fill active half-plane cells only; derive negative-rho ghosts by parity before conversion/constraints |
| IrisK importer/map | imports signed-rho stored cells | interpolate active half-plane cells; parity-fill imported ADM ghosts; authenticate the half-plane map |
| `cartoon_meridional_sampler.hpp` | locator selects signed leaves; central diagnostic averages four rho/z quadrants | accept only physical nonnegative rho; central scalar diagnostic uses the two z sides at positive half-cell rho and the axis-limit reconstruction |
| `cartoon_m0_fastflow.*` | samples a full signed meridian directly | parameterize the complete closed surface with `theta in [0,pi]`, for which `rho=R(theta) sin(theta)>=0`; only the axial z branches are paired for reflection checks |
| `history.cpp` | physical norms give zero measure to negative active cells | integrate every half-plane cell with `2*pi*rho drho dz`; add unweighted axis-tube/off-axis norms and Linf locations |
| PDF output | signed plane is de-duplicated by ignoring rho<=0 | use the same positive-rho cylindrical measure on every active cell |
| restart carrier | signed map/schema and signed central/horizon state | bump schema, preserve active half-plane state and parity-independent horizon/central state, reject legacy signed geometry |
| tests and campaign inputs | domains use x1min=-x1max and often assert mirror refinement | replace with x1min=0 and axis boundary; retain narrow test-only signed references for old-fit comparison |

## AMR semantics

The mesh tree covers only rho>=0.  A leaf touching rho=0 has no negative-rho
neighbor.  Restriction and prolongation act on active physical cells.  If a
coarse/fine stencil crosses the axis, its negative samples are constructed by
the same parity operator on the relevant coarse representation.  A parity fill
follows data motion and prolongation.

Tests require refinement touching the axis, an axis/coarse-fine corner,
different block decompositions, and O2/O4/O6.  They verify parity/restriction
and parity/prolongation commutation to their designed order.  The signed-tree
`ReconcileCartoonRefinementFlags` logic is removed rather than repurposed.

## Horizon, sampling, and integrals

Scalar and tensor sampling accepts a physical `(rho,z)` with `rho>=0`; a
negative rho is outside the half-plane sampling contract rather than an alias
for another point.  The m=0 FastFlow surface remains complete because its
Gauss-Legendre nodes use `theta in [0,pi]` and sample
`rho=R(theta) sin(theta)>=0`, while `z=z_c+R(theta) cos(theta)` covers both
axial sides.  The `mirror_pair` mode pairs horizons displaced along `+z` and
`-z`; it is not a duplicated signed-rho surface.

Physical volume diagnostics use

```text
2*pi * rho * drho * dz * sqrt(det(gamma))
```

for every active half-plane cell.  Coordinate-grid and unweighted axis-tube
norms are reported separately from this physical measure.  Horizon surface
integrals keep their geometric theta weights and are not multiplied by the
volume factor.

Cartoon history records the complete physical-volume global constraint norms,
an unweighted five-cell axis tube, the complementary physical-volume off-axis
region, each local-AMR layer `rho/h=0.5,...,4.5`, and global Linf values with
their `(rho,z)` locations.  Here `h` is the radial spacing of the leaf that owns
the cell, so layer membership continues to mean distance from the axis in
local stencil units on an AMR mesh.

For low-intrusion failure-path diagnostics, native binary outputs with
`variable=z4c` and `variable=con` retain the complete AthenaK post-load state.
`z4c_cartoon_failure_region_extract.py` selects the five historical layers at
`1.0<=abs(z/M)<=1.3`, preserves nonfinite values explicitly, and derives
`K=Khat+2*Theta`, `det(g_tilde)-1`, and `tr_g_tilde(A_tilde)`.  This covers the
requested field, Hamiltonian, momentum, Z, aggregate, and algebraic-constraint
state without adding work to the RHS kernel.  It does not claim term-by-term
RHS decomposition; if the qualified evolution exposes a new causal ambiguity,
that narrower stage-local instrumentation remains a diagnostic follow-up.

## Stability and qualification

The frozen-operator utility compares the legacy fitted closure, the parity
centred half-plane operator, and the production regular-coefficient closure in
a documented six-field radial proxy.  It reports eigenvalues, numerical
abscissa, eigenvector conditioning where meaningful, transient amplification,
RK4 amplification, and mode localization for O2/O4/O6.  Exact tests cover the
full direct-functional inventory separately.  The proxy is representative
linear evidence, not a full Z4c energy proof or nonlinear stability claim.

Physical qualification uses fresh t=0 Kerr punctures with M=1, chi=0.5,
pre-collapsed initial lapse, AthenaK's default advective 1+log lapse and
Gamma-driver shift, O6, RK4, double precision, no chi floor, and finest
spacings M/32, M/48, and M/64 through t=5M.  The gauge contract freezes
`lapse_oplog=2`, `lapse_harmonicf=1`, `lapse_harmonic=0`,
`lapse_advect=1`, `shift_Gamma=1`, `shift_eta=2`, and `shift_advect=1`, with
telegraph and slow-start modifications disabled.  Histories retain
global physical norms, unweighted axis-tube norms, off-axis norms, Linf
locations, and per-layer diagnostics.  Horizon area, irreducible and
Christodoulou masses, spin, radii, center, residuals, and reflection consistency
must converge with the constraints.  A chi=0.99 case starts only after chi=0.5
qualifies.

The legacy side-local fit exists only as immutable baseline bytes loaded by the
stability test through `git show`. It is not compiled into or selectable from
the half-plane production provider. Final legacy-removal status remains
provisional until the half-plane implementation passes all qualification gates.

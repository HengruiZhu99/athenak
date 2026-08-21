# Vertex-centered Z4c architecture

Status: implementation design, schema 1. This document is normative for the
`codex/z4c-vertex-centered-cartoon-amr-20260821` branch. It does not claim that
the vertex-centered path is implemented or qualified until the evidence report
closes every gate.

## Compatibility boundary

AthenaK gains a run-lifetime Z4c sampling choice:

```text
<z4c>/grid_centering = cell    # default
<z4c>/grid_centering = vertex
```

Omitting the key and selecting `cell` must reach the same cell-centered
instantiations and retain the same arithmetic. Hydrodynamics, MHD, radiation,
and their finite-volume storage remain cell centered. Vertex-centered Z4c is a
vacuum feature until an explicit, tested CC-to-VC matter adapter exists;
construction must fail if matter is requested with VC Z4c.

The selected centering is parsed once on the host and stored as immutable
configuration. Host wrappers dispatch to template specializations. A device
cell loop may not branch on centering.

## Layout contract

`Z4cGridLayout` owns all Z4c-native index geometry. Global `RegionIndcs`
semantics remain unchanged.

For each noncollapsed direction with `nx` physical intervals and ghost width
`ng`:

| sampling | active start | active end | active points | stored points |
|---|---:|---:|---:|---:|
| cell | `ng` | `ng + nx - 1` | `nx` | `nx + 2*ng` |
| vertex | `ng` | `ng + nx` | `nx + 1` | `nx + 1 + 2*ng` |

A collapsed direction has exactly one point: start=end=0 and stored count 1.
In particular, Cartoon `nx3=1` never allocates two suppressed-direction
vertices.

Coarse active bounds, coarse stored extents, and the coarse ghost width are
members of the same object. No VC kernel may derive them from cell-centered
`mb_indcs` bounds. Coordinate helpers are distinct:

```text
CellCenterX(q) = xmin + (q + 1/2) (xmax-xmin)/nx
VertexX(q)     = xmin + q         (xmax-xmin)/nx
```

Here `q` is relative to the active start and may be negative in ghost storage.
Topology, not floating-point coordinate comparison, identifies the active
Cartoon axis.

## Storage and dispatch

Every Z4c-native array uses the selected layout: accepted and stage state,
RHS, coarse cache, constraints, Weyl data, telegraph damping, curvature and
admissibility scratch, and AMR provenance scratch. Differently shaped CC and
VC arrays may never alias.

The dispatch shape is:

```cpp
template <typename Centering, typename Symmetry, int FD_STENCIL>
TaskStatus KernelImpl(...);
```

where `Centering` is `CellCenteredZ4c` or `VertexCenteredZ4c`, `Symmetry` is
`Cartesian3D` or `CartoonSO2`, and `FD_STENCIL` is 2, 3, or 4. The existing CC
instantiations remain the regression authority. Layout-aware wrappers cover
ADM/Z4c conversion, RHS, algebraic constraints, ADM constraints, curvature,
admissibility, timestep, AMR sensors, and samplers.

## Vertex topology and degree-of-freedom roles

Each physical vertex has a canonical dyadic integer key. For every
noncollapsed direction,

```text
I_level = logical_lx * nx + local_vertex_offset
I_key   = I_level << (configured_max_level - level)
```

Periodic endpoints are canonicalized before forming the key. All shifts and
multiplications are checked for 64-bit overflow. Collapsed directions use key
zero. Floating-point coordinates are never vertex identities.

Stored points are classified as independent interior, same-level shared,
coarse-fine coincident, hanging fine interface, physical boundary, axis, or
ghost. Hanging vertices are constrained values, not independently evolved
degrees of freedom. The role plan is rebuilt after initial topology creation,
static/dynamic refinement, load balance, and restart.

## Deterministic communication

`MeshBoundaryValuesVC` is separate from `MeshBoundaryValuesCC`. It supports
same-rank and MPI faces, edges, corners, periodicity, physical boundaries,
coarse/fine neighbors, and collapsed directions.

Same-level ranges include the shared active endpoint plus required ghosts.
For a shared physical vertex, contributors are sorted by

```text
(level, logical location, GID, local index)
```

and a single device work item accumulates them in that order and divides by
the topology-derived multiplicity. Floating-point atomics are forbidden. The
same contributor order is used for local and MPI paths. A canonical owner is
only for output de-duplication and diagnostic location reporting, never as an
arbitrary evolution owner.

## Vertex AMR

VC AMR is a separate payload class. Load balance tracks `ncc_tosend`,
`nvc_tosend`, and `nfc_tosend` independently.

- Restriction: exact injection at coincident nodes.
- Derefinement: verify duplicate sibling copies; combine only in deterministic
  order and fail on material disagreement.
- Prolongation: inject coincident nodes; use symmetric midpoint Lagrange
  interpolation for odd fine nodes.

The one-dimensional midpoint rules are `[1,1]/2`, `[-1,9,9,-1]/16`, and the
degree-five-exact symmetric six-point rule for O2/O4/O6. Tensor products are
formed only over active dimensions. Coarse VC ghost width is derived from the
farthest parent needed by active hanging vertices and every allocated fine
ghost target; it is not copied from the CC coarse layout.

The full transfer amplification spectrum is evidence, not an acceptance
threshold. Coincident nodes must never be interpolated.

## Stage ordering and accepted state

Initialization, topology changes, and restarts complete VC synchronization
before an RHS is evaluated. Every RK stage uses:

1. assert prior synchronization;
2. fill axis parity ghosts and verify ghost coverage;
3. check admissibility and calculate RHS;
4. update active independent/shared state;
5. apply active-axis state regularity and check admissibility;
6. inject fine-to-coarse coincident values;
7. exchange and deterministically synchronize shared vertices;
8. fill hanging active vertices, ordinary ghosts, physical/corner data, and
   negative-rho parity ghosts;
9. verify shared equality and admissibility.

At the accepted stage, algebraic projection runs only on independent/shared
active vertices after synchronization. Axis regularity is reapplied and a
second exchange rebuilds all dependent storage from the projected accepted
state. Hanging nodes and arbitrary ghost storage are not projected as
independent values.

## Boundaries and Cartoon axis

A VC endpoint lies on the physical boundary. Outflow/extrapolation,
periodicity, Sommerfeld RHS, and mixed physical/coarse-fine corners must use
that geometry. The inner-rho Cartoon face is an evolved-axis regularity
boundary, never generic outflow. The analytic axis identities and component
parities are specified separately in
`z4c_vertex_cartoon_axis_regularization.md`.

## ADM, consumers, and coupling

VC mode owns a native VC ADM cache for constraints, curvature, Weyl, native
output, and samplers. The existing ADM object becomes a real CC adapter with
owned storage; it must not shallow-slice VC lapse or shift. An explicit,
order-matched VC-to-CC interpolation updates the adapter after each accepted
state.

Every consumer is either made centering aware or rejected with a precise
VC-mode error. This includes wave extraction, CCE, trackers, FastFlow, horizon
dumps, meridional and central observers, and spherical interpolation. The
Brill central observer samples the synchronized `rho=0,z=0` vertex directly.

## Restart, output, history, and replay

Restart records centering, centering schema, active/stored/coarse extents, and
coarse ghost width. A legacy restart lacking these records is cell centered.
Cross-centering restart is rejected unless a separately named offline tool is
used. CC restart bytes remain unchanged.

Native VC binary output carries nodal metadata; VTK uses `POINT_DATA`; slices
use `VertexX`. Formats that cannot represent VC either use an explicit
temporary CC cache or reject the request. They never label VC bytes as CC.

VC history integrals use nodal trapezoidal/dual-volume weights. For Cartoon,

```text
dV = 2*pi*rho * w_rho*w_z * dx_rho*dx_z * sqrt(det(gamma))
```

with endpoint weight one half and interior weight one. The axis ring has zero
volume. `abs(det)` is forbidden as a metric-failure mask. Shared nodes are
counted once using a canonical diagnostic owner.

The AMR event/tree format remains centering independent, while restart and
history provenance include centering/schema. Replaying a CC authority on VC is
an explicitly controlled topology experiment, not a claim that the CC tree is
optimal for VC.

## Qualification boundary

Implementation order follows the phase gates in the controlling goal. No long
Brill campaign is authorized. The bounded discriminator replays the existing
authenticated N256 CC authority only through common central proper time
`tau_c >= 3 M` or an earlier fail-closed state. Exact replay is not numerical
convergence.


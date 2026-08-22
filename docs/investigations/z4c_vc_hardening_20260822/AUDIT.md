# Native vertex-centered Z4c hardening: pre-edit lifecycle audit

Date: 2026-08-22

This document records the mandatory read-only audit performed before changing
production code on `codex/z4c-vc-hardening-2d3d-20260822`.

## Authority and clean baselines

- historical cell-centered authority: `6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`
  (tree `cbb702f4...`)
- native-VC implementation authority: `5d37b5e5c278ac4a1afd52f9553dee6ffed48d0e`
  (tree `e391e288...`)
- evidence handoff: `7e0690e46766cdb101f45aceefe8cc5f2a66ae4f`
  (tree `711cefa9a8f8027fae00f5970e26029be51ba0a8`)
- Kokkos submodule: `6739bc623081648af9e752b616d9671527922cbf`

Separate clean worktrees were created for all three source authorities. A
comparable GNU 13.3 Release MPI+OpenMP build completed in each. Executable
SHA-256 values are `29335a8f...` (CC), `73c448ee...` (starting VC), and
`890ca959...` (candidate). The broad CTest runs were mistakenly concurrent and
oversubscribed the host; their timeout failures are not treated as source
failures. Two static ordering tests fail reproducibly in both starting-VC and
candidate builds because their text parser assumes one unbranched CC/VC task
order. That is an inherited clean red baseline and the tests must be made
centering-aware rather than weakened.

The exact-final Aurora/SYCL evidence remains the governing dynamic failure:
tests 1--16 pass, then `athena.z4c_vc_dynamic_amr` faults on the first accepted
4-to-7-leaf transaction immediately after the retained
`CARTOON_AMR_TRANSACTION` line. The current log does not identify which AMR
phase failed. Perlmutter runtime qualification remains pending.

## Immutable grid geometry

`Z4cGridLayout` is constructed once from the validated centering and the mesh
interval counts. `RegionIndcs` remains cell-centered and is not reinterpreted.
For each non-collapsed direction, CC has `N` active points and VC has `N+1`;
both use the configured ghost width. A collapsed direction is exactly one
stored point at index zero and has no ghosts. The corresponding coarse VC
shape is `N/2+1` active points plus `2*coarse_ng` stored ghosts.

For the 2D dynamic canary (`16x16x1` cells per block, `nghost=4`) native VC
therefore has active `17x17x1`, stored `25x25x1`, coarse active `9x9x1`, and
coarse stored `17x17x1`. `CopyForRefinementVC` copies exactly a `17x17x1`
octant into a `17x17x1` coarse destination. The collapsed direction is
explicitly `[0,0]` in same-rank copy and AMR message metadata.

## Native allocation, aliases, and lifetime

| Object | Owner/allocation | Extent and capacity | Aliases/cached views | Rebuild or rebind event | First/last ordinary-stage consumer | Geometry disposition and coverage |
|---|---|---|---|---|---|---|
| `u0` | `Z4c`, `AllocateNativeStorage` | `[max(nmb_thispack,nmb_maxperrank),25,n3,n2,n1]` | all `z4c.*` tensor slices | backing allocation is fixed after construction; no AMR rebind | `CopyU`/accepted-state conversion and diagnostics | layout-native in 2D Cartesian, 2D Cartoon, 3D Cartesian; host unit/static/fixed-grid coverage, dynamic SYCL red |
| `u1` | `Z4c` | same native shape as `u0` | none | fixed | stage copy/RK update | layout-native; fixed-grid coverage |
| `u_rhs` | `Z4c` | same native shape as `u0` | all `rhs.*` tensor slices | fixed | RHS/KO to RK update | centered compile-time dispatch; 2D Cartesian/Cartoon derivative tests and 3D source/unit coverage |
| `u_con` | `Z4c` | `[capacity,7,n3,n2,n1]` | `con.*` slices | fixed | accepted-state ADM constraints to history/output | native layout; VC ring quadrature and output tests |
| `u_telegraph_mu` | `Z4c` | `[capacity,1,n3,n2,n1]` | direct view | fixed | gauge RHS/history | native layout; no AMR-specific production canary |
| `u_weyl` | `Z4c` | `[capacity,2,n3,n2,n1]` | `weyl.*` slices | fixed | Weyl calculation through boundary/output | native layout; output/static multilevel coverage |
| `u_adm_native` | `Z4c`, VC only | `[capacity,ADM::nadm,n3,n2,n1]` | native ADM tensor slices | fixed | Z4c-to-ADM to VC diagnostic/output samplers | explicit native VC cache; legacy CC consumers use the separate CC adapter |
| `coarse_u0` | `Z4c`, multilevel | `[capacity,25,cn3,cn2,cn1]` | direct view | fixed | restriction/copy/receive to prolongation | layout-native; dynamic AMR is the unqualified production path |
| `coarse_u_weyl` | `Z4c`, multilevel | `[capacity,2,cn3,cn2,cn1]` | direct view | fixed | Weyl restriction to prolongation | layout-native; static multilevel coverage |
| `chi_provenance_terms` | `Z4c`, default-off diagnostic | native shape when enabled | direct view | fixed | RHS checkpoint diagnostic only | currently rejected for VC; no accidental CC interpretation |
| `MeshBoundaryValuesVC` objects | `Z4c` constructor | buffers sized from immutable `VertexBoundaryLayout` and maximum pack capacity | buffer metadata and request arrays | neighbor records are read from current `pmb`; objects persist across AMR | init-receive to clear/unpack/prolong | separate VC index constructor; collapsed counts are one; static boundary-index tests |
| `Z4cVertexTopologyPlan::records` | topology plan | `[current_local_nmb,n3,n2,n1]` DualView | fetched at each VC prolongation call | **must be reallocated/rebuilt after new `MeshBlock` and `SetNeighbors`** | shared-node sync/prolong role query | rebuilt after initial construction, restart ownership changes, and every accepted AMR transaction |
| topology contributor tables | topology plan | host vectors over current contributors; device `local_indices[local_count,4]` | shared-node synchronization | **must rebuild after rank ownership/topology changes** | shared-node gather to canonical scatter verification | deterministic global key order; MPI topology tests, dynamic device lifecycle unqualified |
| `newtoold`/`oldtonew` | `MeshRefinement`, per transaction | exact new/old global leaf counts | host arrays; copied into local `new_to_old` DualView | deleted at transaction end | copy/AMR metadata to refine kernel | centering-independent maps; all device closures must capture device view only |
| AMR `sendbuf`/`recvbuf` | `MeshRefinement`, MPI builds | number of migrating blocks; fixed `send_data`/`recv_data` capacity | metadata DualViews plus MPI requests | metadata reallocated each transaction; request lifetime ends at clear | pack to unpack | CC/VC/FC counts separated; VC inclusive ranges use `cntvc` |

Because evolved arrays are allocated once at maximum MeshBlock capacity, the
shallow AthenaTensor slices remain bound across AMR. This does **not** imply
that topology records, local contributor indices, neighbor views, or request
metadata remain valid; those have a distinct topology/index lifetime.

## Accepted AMR lifecycle and ownership

| Phase | Operation | State/lifetime disposition before instrumentation |
|---|---|---|
| A0 | entry to redistribution | old mesh metadata and old topology plan authoritative |
| A1 | build logical locations and `newtoold`/`oldtonew` | exact global map extents required |
| A2 | compute load balance | fixed native allocation checked against new local occupancy |
| A3--A4 | initialize receives; pack/send | old `pmb`/neighbors and old array slots authoritative |
| A5 | VC derefine | deterministic sibling aggregation; split-rank families use receive path |
| A6 | move retained VC blocks | same backing allocation; ordered left then right logically |
| A7 | copy refined-parent octants | fills every new child coarse cache from the moved parent slot |
| A8 | complete MPI and unpack | new-slot active/coarse data become authoritative |
| A9 | synchronize `new_to_old` device view | refine kernel must see current maps and flags |
| A10 | `RefineVC` | fills new-child active points and validates finite/positive chi |
| A11--A13 | install mesh metadata, replace block/coordinate objects, set neighbors | old block/coordinate/neighbor views become invalid |
| A14 | rebuild topology plan | records and contributor indices must match new ownership and new `pmb` |
| A15 | algebraic projection | acts on new active VC state only |
| A16 | boundary initialization | rebuilds restriction, shared-node exchange, BC/parity, and CF prolongation |
| A17--A19 | ADM conversion, constraints, timestep | first downstream scientific consumers of accepted state |

## Geometry-use dispositions

- `z4c_grid.hpp`, native allocation, RK update, RHS dispatch, ADM conversion,
  axis treatment, VC restriction/prolongation, VC boundary communication,
  topology records, restart, and native output loops use `Z4cGridLayout`:
  **layout-native and accepted**.
- `RegionIndcs` in mesh-tree construction, rank assignment, physical MeshBlock
  bounds, AMR cadence/history, and generic hydro/MHD/CC paths describes the
  underlying cell mesh rather than native VC state: **accepted infrastructure
  geometry**.
- `RegionIndcs` inside explicit VC-to-CC adapter or sampling paths (legacy ADM
  consumers, formatted cell outputs, meridional samplers): **accepted only
  because the path constructs or samples an explicit CC adapter**.
- `CellCenterX` on native VC state is rejected. Native pgens, derivative/RHS,
  ADM, and Weyl coordinates select `VertexX`; `z4c_linear_wave` branches on
  centering. Remaining `CellCenterX` references are CC adapters or legacy CC
  consumers.
- `VertexX` on CC state is rejected. No unconditional CC production path uses
  it.
- The controlling prompt names `src/z4c/adm_z4c.cpp`; the repository file is
  `src/z4c/z4c_adm.cpp`, which was audited in its place.
- Three-dimensional Cartoon remains rejected by configuration. Supported
  geometries are 2D Cartesian VC, 2D axis-touching Cartoon VC, and 3D
  Cartesian VC.

## Communication/count audit

Native VC variables are excluded from `ncc_tosend` and included once in
`nvc_tosend`. The total message span is
`ncc*cntcc + nvc*cntvc + nfc*cntfc`; VC pack and unpack use the corresponding
offset and flatten precisely the inclusive VC bounds. Same-level messages use
`(N+1)` active points; refined-parent messages use the complete coarse stored
shape; derefine messages use `(N/2+1)` points. Collapsed dimensions contribute
one, not a ghosted nominal width.

The current code derives `cntvc` from the same inclusive endpoints used by
pack/unpack, but it has no diagnostic postcondition proving that each cursor
ends at its declared span or that matching sender/receiver metadata agree.
Those checks are required in Phase 2.

## Execution-space and lifetime risks requiring observation

1. `CopyCC`/`CopyVC`, `CopyForRefinementVC`, AMR pack/unpack, and `RefineVC`
   repeatedly construct `DevExeSpace()` rather than carrying one explicit
   execution-space instance through the producer/consumer chain. The source has
   no explicit dependency from same-rank subview copies to the consuming
   refinement kernel. CUDA default-stream behavior is not proof of SYCL queue
   ordering. **Hypothesis only:** a missing cross-instance dependency could let
   SYCL consume an incomplete coarse cache or destroy/rebuild objects while work
   remains outstanding.
2. `RefineVC` does write its returned interpolation value: the write occurs
   inside `vertex_amr::ProlongVCPoint`. A suspected missing destination store was
   explicitly ruled out.
3. The topology plan is rebuilt after `SetNeighbors`, and boundary code fetches
   `records.d_view` at call time. No static stale-record capture was found.
   Outstanding asynchronous consumers at the moment of `records` reallocation
   remain to be excluded dynamically.
4. Native arrays do not reallocate during topology replacement, so no shallow
   Z4c/ADM/Weyl tensor needs rebinding. The MeshBlock and Coordinates objects do
   get destroyed and reconstructed; any outstanding kernel that captured their
   views would be a use-after-free.
5. Exact-final Aurora evidence ends after the transaction summary and before a
   named completion point. It does not prove that `RefineVC`, topology rebuild,
   boundary initialization, or the first downstream diagnostic caused the page
   fault.

## Existing production coverage gaps

- No A0--A19 backend-portable completion record exists.
- No contiguous first-error record validates AMR map indices, subview bounds,
  message spans, or topology generation on device.
- The dynamic fixture is 2D Cartoon only; production-path dynamic tests are
  missing for 2D Cartesian and 3D Cartesian.
- The exact-final SYCL run fails before dynamic AMR qualification; Perlmutter
  CUDA runtime evidence for the exact final source is absent.
- Static task-order tests currently conflate branched CC and VC task graphs.
- Host unit tests cover formulas and isolated operators more strongly than the
  complete production AMR/load-balance/object-replacement lifecycle.

## Phase-1 disposition

No root cause is established by source inspection alone. The next permitted
action is default-off instrumentation that fences and validates each A0--A19
phase, followed by one exact Aurora canary retry per recorded hypothesis. A
permanent fence, transfer change, or lifecycle repair is not justified until
the first bad phase and invariant are demonstrated.

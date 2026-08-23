# Native-VC Cartoon axis ghost-fill audit

Date: 2026-08-23

## Exact mirror contract

For an active VC axis index `ia`, ghost depth `d=0..ng-1` uses

```text
target = ia - d - 1
source = ia + d + 1.
```

Equivalently, the first negative-rho ghost at `-h` is copied from the active
`+h` vertex, not from the active axis vertex. The packed-component parity is
then applied. The centering-aware implementation is
`FillCenteredAxisGhostLine<Centering>` in
`src/z4c/cartoon_axis_boundary.hpp`.

## Repaired defect

`ReconstructConstraintAxisParityGhosts` previously used cell-centered layout
indices and the CC mirror map on a native VC constraint array. It now uses
`layout.is`, `layout.n2`, `layout.n3`, and
`FillCenteredConstraintAxisGhostLine<VertexCenteredZ4c>`. This is a real
off-by-one geometry defect: the former first VC ghost could be sourced from
the active axis value rather than the `+h` partner.

Disposition: `DEFECT_FOUND_AND_REPAIRED`.

## Production-path inventory

| Data family | Production writer/fill path | Centering-aware status |
|---|---|---|
| evolved `u0` (all 25 variables) | `Z4c::ReconstructAxisParityGhosts`; physical completion in `MeshBoundaryValues::Z4cBCs` | VC helper used |
| stage RHS | axis regularity imposed before and after KO by `ApplyVertexAxisRegularity`; RHS itself is not a communicated state halo | exact active-axis identities imposed |
| `coarse_u0` | physical completion in `MeshBoundaryValues::Z4cBCs` using coarse layout bounds | same centering-aware helper |
| native ADM metric/K | IrisK and Kerr problem-generator reconstruction paths | `FillCenteredAdmAxisGhostLine<Centering>` |
| seven constraints | `ReconstructConstraintAxisParityGhosts` | repaired VC helper used |
| curvature/Weyl scratch | reconstructed from parity-complete native ADM/Z4c fields through the common derivative provider | no independent handwritten VC mirror |
| restart state | restart load enters the ordinary receive/physical-BC/parity/finalize task chain | same production helper |
| diagnostic sampler | reads post-boundary state/RHS/constraints; canonical shared-node ownership is explicit | no ghost writer |

## Stage and corner order

The task sequence establishes one deterministic contract:

```text
copy synchronized stage state
-> rebuild negative-rho parity ghosts
-> compute bulk RHS
-> apply physical Sommerfeld RHS
-> RK update
-> synchronize/restrict/receive
-> fill physical z/rho ghosts
-> prolongate if multilevel
-> rebuild negative-rho parity ghosts from completed positive-rho data
-> project accepted active VC state
-> rebuild communication/boundary state once more.
```

At an axis/z-boundary corner, physical z completion therefore precedes the
final negative-rho mirror. The negative-rho data are derived from completed
positive-rho values and never become an independent authority.

## Poison and lifecycle tests

`tst/unit/z4c/cartoon_vertex_axis_test.cpp` initializes all negative-rho
ghosts with NaNs, invokes the real device helper, verifies every ghost against
`parity*positive_source`, and verifies active values are unchanged. It covers:

- all 25 packed Z4c components;
- all native ADM components;
- all seven constraint components;
- `ng=2,3,4` (O2/O4/O6);
- multiple MeshBlocks;
- scalar, vector, symmetric-tensor, ADM, constraint, and RHS regularity.

The fixed-grid CUDA gate additionally passed Cartoon restart and output tests.
The current-source host subset passes the poison, restart, CC exact-fingerprint,
and production-kernel regressions. Current-source SYCL execution was not
available and is recorded as a limitation.

## Remaining qualification boundary

Bitwise agreement of shared native vertices and exact axis mirror filling do
not establish outer physical-boundary convergence. The residual fixed-grid
failure is localized first to the outer face, while a later negative-order
`B_x2` result (reported as `By`, the active-z component) lies at the intersection of the axis and a
same-level `z=4` MeshBlock seam. That later seam-axis observation is not
attributed to ghost filling without a dedicated writer/derivative replay.

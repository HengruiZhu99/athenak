# Axisymmetric Cartoon Z4c

AthenaK's Z4c module accepts a host-selected symmetry mode:

```text
<z4c>
symmetry = cartesian3d
```

`cartesian3d` is the default. It retains the existing coordinate map and direct generated
finite-difference operations. `cartoon_so2` selects the signed meridional map
`(x1,x2,x3)=(rho,z,suppressed Y)`, or physical Cartesian
`(X,Y,Z)=(x1,x3,x2)`. Host code selects a separately compiled symmetry/stencil pair;
device cell loops must not capture or branch on the runtime symmetry mode.

## Current staged availability

The common dispatch, fail-fast configuration, restart carrier, collapsed stored bounds, PDF
policy, and shared Z4c derivative kernels are present. The common RHS (including Gamma, gauge,
and dissipation), constraints, fixed-second-order Sommerfeld condition, curvature diagnostics,
derived/history curvature consumers, and Weyl calculation are separately instantiated for all
three supported stencils. The migration accounts for all 61 original logical raw-FD sites: 58
are owned by the shared policy (one ADM initialization site arrived in the earlier storage
slice), while the three remaining live raw calls are confined to the Cartesian-only dynamical
GRMHD matter path.

AMR, axis interpolation/history, initial data, and horizon integration are not all integrated
or qualified yet. Consequently, this milestone still rejects every `cartoon_so2` problem
generator before allocation. The later Cartoon Kerr adapter slice will enable its pgen only
after its coordinate/component map and collapsed storage have focused tests. Do not interpret
the public mode parser or compiled kernel readiness as a qualified evolution backend.

The existing output lifecycle writes the initial history and derived-variable records before
ADM scratch is initialized. Consequently, `maxAbsKret` is `inf` and cellwise curvature values
are `NaN` in the `t=0` records even in the pre-migration Cartesian executable; records emitted
after initialization are finite. Qualification must either move/guard that initial diagnostic
or treat the initial record as unavailable. Kernel migration tests retain an explicit sentinel
for this limitation and must not be read as claiming that every output record is finite.

## Preallocation contract

Before units or any physics object is constructed, `cartoon_so2` requires:

- vacuum Z4c with no hydro, MHD, ion-neutral, radiation, turbulence driver, particles, or
  dynamical matter source;
- global and MeshBlock `nx3=1`, an active `x1-x2` plane, positive even global `nx1`, an even
  number of root x1 MeshBlocks, and finite symmetric `x1min=-x1max` so the internal axis lies
  between cell centers and no root block straddles it;
- `coordinate_map=signed_rho_z_suppressed_y_v1` when the map is specified, and
  `symmetry_schema=1` when the schema is specified;
- an effective spatial order of 2, 4, or 6 with enough ghost cells for stencil widths 2, 3,
  or 4; as in the existing Cartesian path, a requested order less than or equal to zero uses
  `2*(mesh/nghost-1)`, so `nghost=2,3,4` select orders 2, 4, and 6 respectively;
- no compact-object tracker, Cartesian wave extraction, CCE extraction, Cartesian horizon dump,
  legacy tracker-dependent FastFlow option, or pre-m=0 FastFlow construction;
- only `tab`, `hst`, `log`, `vtk`, unweighted `pdf`, `bin`, and `rst` output types; and
- compatible restart symmetry, coordinate-map, and schema metadata when those carrier fields are
  present.

`cart`, `sph`, `cbin`, `pvtk`, and `trk` outputs are rejected before wrapper construction.
Unknown output tokens are rejected with the complete supported allowlist. Cartoon PDF output
rejects `mass_weighted=true`: vacuum Z4c component zero is conformal factor `chi`, not density.
Without `variable_2`, every second-axis PDF key is forbidden. With `variable_2`, explicit
`nbin2>=1` is required; the later PDF slice adds the complete checked count/bounds/allocation
contract.

Restart-origin metadata is made immutable against `-i` and command-line overrides by the
`<z4c_restart>` carrier. The carrier is internal and cannot be supplied on a fresh start. It
records the symmetry mode, coordinate map and schema, requested and effective spatial order,
stencil width, axis-central proper-time integration state, and reserved m=0 FastFlow surface
state. It is captured from the restart parameter dump before `-i` or command-line processing;
every conflicting override reports the block, key, stored value, and requested value before
mesh or physics construction. Compatible values are restored from that immutable snapshot.

The carrier uses schema 1 and does not alter the binary restart layout. Legacy Cartesian
restarts without the block retain their previous behavior and acquire a carrier when next
written. The central sampler and m=0 FastFlow algorithms are later slices, so their state is
currently initialized to explicit inactive defaults. No Cartoon pgen is enabled by this
carrier slice.

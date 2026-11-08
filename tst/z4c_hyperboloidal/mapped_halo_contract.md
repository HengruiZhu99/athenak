# Native mapped halo contract

`NativeDonorSelector` reads the native `Mesh` leaf locations and atlas. Its
metadata contains maps and native global block IDs, without field storage or a
second ownership tree. Local ownership is always the native `MeshBlockPack` GID
range. All interpolation donors are independently evolved cells, including the
explicit overlap cells. Ghosts are never donors.

The six-point selection preserves the N12 reference policies: finest eligible
Cartesian core leaf, aligned direct copies, centered evolved-overlap selection,
and physical-radius shell chains. Each shell radial plane uses its own angular
leaf when refinement changes the angular partition. Cartesian tensor components
remain in their common basis. Geometry tie ordering is independent of rank
ownership. Coalesced accumulation order follows native GID/index order, so
bitwise equality with the old block order is not assumed.

`MappedHalo::Rebuild(pack, origin)` constructs only the union of the bulk
radius-four axis KO stencils and radius-two mixed derivative stencils. It
classifies every owned evolved cell into one of four disjoint lists:

- `InteriorPoints`: all stencil inputs already owned evolved cells.
- `LocalHaloPoints`: additionally needs interpolation using local donors only.
- `RemoteHaloPoints`: needs at least one remote interpolation donor.
- `SurfaceDependentPoints`: needs a scri/exterior radial target, possibly also
  local or remote donors. This category takes precedence over remote dependency.

Each exterior target is returned with its physical location and is left
untouched. Generic interpolation never substitutes zero or reference data for
missing stage-dependent scri closure. Unused padding stays poisoned in tests.
Additional surface and physical angular-dissipation stencils must provide their
own dependency plans; these lists describe the stated bulk stencil union only.

Two persistent `bvals::MappedExchange` instances share one donor topology:
26 independent reference-subtracted residuals and all 31 raw physical fields.
The residual supplied by the module already divides the chi difference by
radius squared. Both outputs use their own donor values and identical weights;
the physical interpolation is neither reference-added residual interpolation nor
algebraically reprojected state. Combining these payloads is a later transport
optimization, not a change to the numerical contract.

Stage ordering is `StartReceives`, `PackAndSend(residual,physical)`, then
`FillLocal` while `Receive` polls. `FillRemote` requires successful receive
completion. Both clear tasks poll without blocking. Rebuild is the only
collective planning phase. One-rank geometry failure is propagated to all ranks
before transport metadata collectives, and leaves the wrapper unusable until a
successful rebuild. Native exchange generation checks reject stale plans after
load balance or AMR changes; rebuilding does not itself migrate field state.

The reproducible oracle runner is `mapped_halo_oracle.py` with `--build`,
`--legacy`, `--output`, and `--ranks`. It links native AthenaK objects and the
current mapped-halo source. It creates a private legacy-header overlay which
suppresses exactly the constructor's eager `BuildHalos` call, retaining the
unmodified legacy interpolation selector. This avoids allocating global legacy
halo plans just to query selected reference rows. The original and overlay
hashes and the exact change are recorded in `overlay-provenance.json`.

N12 tests compare sampled rows by default; `--full-rows` checks all 481248
required halo rows against the legacy selector, including the refined Cartesian
core. Three non-polynomial payloads also check anchored interpolation against the
old padded-block summation order on each tested row. Every native raw/residual
filled value is checked over two changing stages. The predeclared donor-weight tolerance is
`1e-12 + 5e-10*abs(reference)`; the fill tolerance is absolute `1e-12`.
`--shell-smr` uses the native 197-leaf fixture and checks reproduction of
radial degree-zero through degree-five polynomials multiplied by angular
polynomials, plus the same complete transport/fill checks. This proves the
static donor plan properties exercised there, not AMR field migration, complete
scri evolution, or an evolving PDE trajectory.

Neither selector nor exchange changes native load-balancing weights: all leaves
retain cost one, including refined core and shell leaves. SMR topology and donor
selection are supported by the tested fixtures; moving existing evolved state
between changing AMR topologies remains a separate requirement.

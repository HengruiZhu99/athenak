# Complete N12 stage RHS oracle

`coupled_rhs_oracle.py` links the original native AthenaK executable's objects,
replacing only its `main` with the regression entry point. All native stage
plans and RHS implementations must already belong to that executable. The test
does not implement a second evolution driver or advance an RK stage.

The test constructs the pinned 106-leaf N12 topology through native `Mesh`,
`MeshBlockPack`, and the hyperboloidal problem generator. It invokes actual
module stage wrappers, including algebraic reconstruction, outgoing surface
projection, residual/raw donor transport, local and remote interpolation,
normal compatibility, scri RHS, all four disjoint volume RHS batches, physical
KO, and request retirement. Before each of two scenarios, bulk ghosts,
residuals, and both RHS arrays are poisoned with NaNs.

The independent reference is the read-only legacy `SphericalTransfer3D` CPU
model. Its reference N12 mesh is constructed independently and mapped to native
leaf GIDs using chart/layer and physical block bounds. Every bulk sample
coordinate is required to be bitwise equal before rates are compared. The
reference uses KO 0.1, penalty 0.5, the native pinned timestep and gauge
parameters, plus its explicit experimental-timestep switch to avoid introducing
CPU-only CFL/penalty rejection absent from the campaign's GPU operator. That
switch changes admission checks, not computed rates.

Scenario zero uses the spherical analytic initial data. Scenario one adds a
smooth nonspherical perturbation to every independently evolved field, with the
chi perturbation multiplied by radius squared. Both scenarios reconstruct
dependent metric/curvature fields before comparing complete input states.
Native and reference inputs are generated at their own coordinates; reference
arrays are not copied into native state to suppress a mismatch.

Every one of the 31 stored fields is compared at all 229392 evolved bulk points
and 864 separately evolved scri points. The previously adopted short-operator
gate remains `abs(delta) <= 1e-12 + 5e-10*abs(reference)`, with chi using the
tighter absolute term `1e-14`. Failure output includes scenario, rank, native
GID, cell indices, field, values and tolerance, plus per-field counts. The
runner records source/executable hashes and rejects concurrent source changes.

The error RMS uses the reference `grid.volumes` on every independently evolved
bulk point, including explicit overlap and the endpoint-grid half weight at the
first outer-layer radial sample. It is a norm of the complete evolved state,
not a physical diagnostic integral that removes overlap using partition-of-unity
weights. Surface errors use area weights obtained from the reference physical
cell volume at scri divided by its radial spacing. Each reported RMS is
`sqrt(sum(weight*delta^2)/sum(weight))`; category aggregates include all 31 fields
and both scenarios, while per-field RMS values are also recorded. Machine JSON
includes all per-field absolute/scaled maxima and the rank, scenario, native
GID, indices, physical coordinates, and values at both the absolute and scaled
worst locations.

This test proves a complete stage RHS for its two input states when it passes.
It does not prove multi-step integration, AMR state transfer/migration, restart,
Aurora GPU correctness/performance, or agreement with the saved 5M trajectory.

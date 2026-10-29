# Controlled cross-face scri angular operator

The original n16 Aurora Minkowski control failed at coordinate t=0.4895.
Its growing momentum, Z4 and tensor-constraint errors concentrate at the
corners of the actual scri atlas. The volume grid exchanges ghost data across
faces, but the original scri jet reconstruction clamps its six angular nodes
inside each face. Thus the boundary atlas does not receive neighboring-face
surface values through that derivative operator. This is a concrete interface
gap and a candidate cause; the failure localization alone is not a proof.

The opt-in `cross_face_scri` argument of `HyperbolicComposite` changes only the
scri derivative reconstruction. Each angular direction uses five centered
nodes. A node outside its destination face obtains values from the physical
owner: existing six-point active-volume interpolation for interior radial
nodes, and six-by-six interpolation on the actual surface for the endpoint.
All components remain in the global Cartesian basis. Destination-map Jacobian
and Hessian terms convert the resulting chart derivatives into Cartesian jets.
The five interior radial nodes plus actual scri endpoint remain unchanged.

Composed donor weights are flattened and merged into the existing term table.
All volume donors are active cells; no ghost or face-order dependency is added.
The row's actual surface point is the subtraction anchor, preserving constants
exactly. The stencil neither clips values nor projects interpolated tensors.
Its consistency does not establish an energy estimate or stable evolution.

`hyperbolic_shift_3d_mpi_projected_angular` enables this operator together with
the separately tested standard final-stage algebraic repair. Its checkpoint
magic is `HYP3D_MPI_V3`. The projection-only target uses V2 and the original
control uses V1; cross-scheme restarts reject. Reports record the angular
choice explicitly. This permits a fresh V3 comparison against the V2 trajectory
without silently changing a resumed scheme.

The manufactured nonspherical scalar test covers n6/8/12/16, angular interiors,
edges, corners, mixed derivatives, exact constants, active-only donors, and
actual cross-face dependencies. Fine-grid corner gradient/Hessian convergence
orders are 4.51/4.76. The n16 flattened table has 315,784 terms, at most 617 per
row (35.37 MB host storage). Full and split native CPU/device-path results match
exactly on the local Serial backend for Minkowski and a nonspherical seed.
These are component consistency checks, not resolution convergence of the PDE.

[Local MPI evidence](reproduction/mpi-angular-local-v1/assessment.json) covers
one Minkowski step and four pulse steps at n8. Direct, restarted and two-rank
checkpoints match bitwise across all four arrays. V2 rejects a V3 restart.
Null closure remains below 9e-16 in these short tests. No Aurora GPU or long-time
stability result is claimed here; those require independent fresh campaigns.

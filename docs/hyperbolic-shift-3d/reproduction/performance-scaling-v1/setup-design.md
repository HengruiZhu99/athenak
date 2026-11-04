# M4: construct the final halo plan once

M0 measured host-model setup at 15.6–18.8 seconds per generation on the initial
single-rank dynamic-wave qualification, versus 0.2–1.4 seconds for each next
setup subdivision. The Evolution constructor creates HyperbolicComposite, whose
CompositeDifference builds all halo and extrapolated-halo rows with the default
inner-transition flag. Evolution then calls SetInnerTransitionLogical(true),
which discards and rebuilds those rows. No stage uses the first rows.

Add an optional final inner-transition flag to both host constructors, defaulting
to false for source compatibility. The opt-in AMR setup target passes the already
selected numerical flag during construction. The existing setter remains; it is
now a no-op in that target. There is no owner index, interpolation rewrite,
coordinate-map pruning, geometry cache, reordering or transfer arithmetic change.
The generic native AthenaK library does not include either changed header (its
compiler dependency files were checked), so the standalone experiment may link
the frozen native library and replay the original compile/link commands.

The flag changes only CompositeDifference::Interpolate's inner-shell selector.
Geometry, active ordering and Cartesian owner/overlap selection are independent
of it. BuildHalos only replaces its own row vectors. ScriSurface and the direct
radial plan use outer-layer donors; their complete donor order and derivative
weights are compared explicitly to rule out a changed construction dependency.
All six layers, the four ghost layers, six-point halo interpolation, native
NGHOST4 transfer, equations and AMR1 checkpoint semantics stay unchanged.

The focused contract test uses real native 8/15/71/22-leaf descriptors, including
coarsening back to 15 leaves before reaching 22. It compares active/inverse maps,
logical cells, geometry coordinates/Jacobians/Hessians, volumes, every halo and
extrapolation donor/weight, scri state/rows, and every radial derivative row.
Its old-then-new construction times are diagnostic only; accepted timing comes
from independent whole-driver alternating pairs. Full AMR1 wave, equilibrium,
boost, rejection and changed-rank restart checks remain required.

Construction still finishes collectively before publishing the new Evolution.
No validation, safety check, ownership guard, capacity bound, rollback path or
failure capture is removed. This avoids an unused setup pass; it is not an
adaptivity/admissibility fix or a claim of asymptotically local memory.

# Native scri AMR transfer

`SurfaceRemesh::Prepare(old_module, layout)` captures only the old local owned
surface state and fetches the sparse remote donors needed by local new faces.
The layout contains native old/new GIDs, owners, maps and ancestry metadata;
it never contains global field state. `Install(new_module)` runs after native
pack/maps and module storage have been rebuilt. It validates local face order
and copies the prepared local state and records into the new module.

Unchanged outer leaves copy all 31 fields and all eight record members without
arithmetic, even when GIDs and owners change. Changed outer leaves interpolate
26 independent raw Cartesian components using a six-by-six angular tensor
stencil on the containing old outer patch in the replacement family. Point
coordinates use the actual target forward map and old inverse map. The stencil
clamps to that old patch; it does not import another chart's independently
evolved data or invent faces on inner radial siblings. Dependent components 6
and 12 are reconstructed and dormant components 28–30 are zeroed. The ordinary
native initialization path subsequently applies the existing scri projection,
regenerates halos and solves normal compatibility.

Each old surface ChiRecord is an indivisible history packet. It routes once to
the nearest replacement-family angular sample in chart coordinates, with exact
index ties choosing the lower index, followed by new-GID/k/j ordering. Candidate
distances are normalized by the old patch span. A target copies its first packet
and merges later packets in old-GID/k/j order: counts and accumulated correction
add, minima and first-event time minimize, maximum correction maximizes. Targets
with no ancestors receive the explicit ChiRecord defaults. This policy neither
interpolates event counts nor multiplies them through refinement.

The test runner links the actual native objects. It covers 106↔113-leaf outer
refinement/coarsening and a 344-leaf central-refinement fixture that shifts
unchanged surfaces across the native equal-weight partition. The same test
executable passed the reviewed admission on two and eight local MPI ranks.
All unchanged fields and records, each history packet's unique coverage, and
global ledger totals/extrema matched exactly. The eight-rank migration moved
576 surface points and exercised three old and six new ranks with no surfaces.

The original coarse smooth-rate gate failed and remains recorded. For
`sin(5.1u) cos(4.7v)`, N12→24 orders were 5.471545 for refinement and 5.347359 for
coarsening. Independent high-precision interpolation reproduces the refinement
error, and every actual point at N12/24/48/96/192 satisfies the analytic tensor
remainder bound plus the fixed roundoff allowance. Actual N48→96→192 orders
were 5.976415 and 6.034427 for refinement, and 6.012040 and 6.018994 for coarsening.
All supported degree-five tensor polynomials passed the original roundoff gate.

The [scientific review](amr-interpolation-review.md) explains the revised
admission without changing the numerical operator or original failed threshold.
[Compact evidence](evidence/native-surface-remesh-validation.json) reports
`original_gate_passed=false` and `revised_admission_passed=true` separately.
This transfer-only evidence does not establish long-time coupled AMR stability
or GPU performance.

Reproduce the reviewed tests with `surface_remesh.py --build build-native
--output build-native/surface-remesh-reviewed-two --ranks 2 --reviewed-admission`,
then use a new output directory, `--ranks 8`, and
`--reuse-executable build-native/surface-remesh-reviewed-two/surface_remesh` to
run the exact same executable. The default invocation retains the original
coarse-rate gate; `--diagnostic` preserves its overall failure while executing
all remaining transfer checks.

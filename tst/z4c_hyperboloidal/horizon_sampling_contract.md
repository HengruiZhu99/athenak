# Native sparse horizon sampler oracle

Declared before the first sampler comparison on 2026-11-08. This tests the
physical ADM sampling layer, not the FastFlow iteration or a PDE trajectory.

The independent oracle is the exact `EndpointHorizonSampler` class extracted
from the read-only predecessor's
`tst/hyperbolic_shift_3d/endpoint_policy/fastflow_history.hpp`. Only that class
is extracted: the old root-only FastFlow/evolution wrapper is excluded. The
runner hashes the original helper, exact extracted bytes, actual compiler
dependencies, all native source headers/translation units, and linked native
objects. The same executable is reused for 2- and 8-rank validation. A reused
binary must match its original source/dependency/object hashes before running.

All native leaves and all 229392 bulk input points are matched to the old N12
geometry. Native/old coordinates must be exactly equal. The test distributes
old input fields to their native owners; this input/oracle scaffolding is not
part of runtime sampling. The runtime may stage only queried tensor nodes and
their evolved sparse donors, never gather full state.

Three independent inputs are compared: fresh initial reference, a prescribed
nonspherical perturbation of every independent field, and the pinned saved
step3551 fields (SHA256
`7d37c940eea84a04dc471bb90178c3e2facf153f14382cd2a47cf1fba6a320be`). The saved
accepted state is copied without projection or materialization. Native residuals
are formed from those exact values, including the reference's sqrt(r2)-squared
chi normalization.

Queries sample two near-horizon spheres, both sides of central child seams,
triple-corner tensor stencils, core/shell and radial-layer seams, all six shell
faces, angular face corners, and invalid/outside/scri/puncture points. Compare
unique owning rank to the old geometric owner and require identical validity.
An owned invalid point must retain owner=1 with invalid geometry; an unowned
point must return owner=0. No angular average or spherical reduction is used.

For all 45 physical Cartesian ADM components (nine metric, nine extrinsic
curvature, and 27 first metric derivatives), use the adopted short arithmetic
gate `abs(delta) <= 1e-12 + 5e-10*abs(reference)`. Report per-component maxima,
scaled errors, equal-query-weight RMS, worst tolerance ratio and coordinates,
and full sample counts. This RMS is an error diagnostic over prescribed
queries, not a volume norm. Exact agreement is reported when obtained; the
threshold is not adjusted after inspecting errors.

All non-evolved `u0` and residual padding starts as NaN. Thus full tensor corner
requirements cannot be satisfied accidentally by an evolution halo. Require
actual ghost-node and remote-donor use, and independently enumerate a positive
number of triple-corner nodes in valid queries. Compare the full local `u0` storage,
including poisoned ghosts, bytewise before/after sampling. Repeat an identical
batch without exchanging donor values again and require bitwise identical
results. All three scenarios reuse one sampler, exercising state-version cache
invalidation. Each scenario begins with one query, forcing the other ranks to
participate in sparse exchange with no owned samples. Complete small query
outputs are saved for a bytewise 2/8-rank comparison; these contain no evolution
field arrays. Additional restart/remesh and full diagnostics immutability
coverage belongs to the coordinated diagnostics lifecycle tests.

The 2026-11-08 cache-safety extension additionally selects an actual nonzero
ghost-stencil dependency whose evolved donor owner differs from the query
owner. Only that donor owner changes one trace value, invalidates diagnostics,
and reforms its residual. Every other rank's StateVersion must remain
unchanged. The existing sampler must then agree bytewise with a newly
constructed sampler on all queries, and the selected remote query must differ
from its pre-mutation value. This checks cross-rank cache invalidation rather
than only uniform lifecycle invalidations. Earlier successful artifacts under
`build-native/horizon-sampling-{two,eight}` retain their original hashes and do
not claim this new cache-safety regression.

Build/run through `horizon_sampling_oracle.py` against the original native
AthenaK build; all native runtime objects are linked unchanged. Compile with
`-ffp-contract=off`, use `make -j64`, and serialize the whole native build with
other agents. Correctness runtimes are not performance measurements. Separate
2/8-rank results are mandatory; one decomposition does not establish the other.

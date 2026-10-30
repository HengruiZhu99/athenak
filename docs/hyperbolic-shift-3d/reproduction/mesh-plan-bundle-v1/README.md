# Coordinated host/device mesh-plan lifetime

`HyperbolicMeshPlans` owns one complete generation of host geometry, radial and
angular plans, device operators, halo plans, MPI ownership/counts, sparse donor
exchange, and reference lapse. It supports the sampled-normal, donor-seven
formulation. The library was first tested independently. It is now used by the
opt-in [planned evolution target](../mesh-plan-evolution-v1/README.md), including
fixed-layout accepted-step rebuilds. The measured V24 target retains its existing
independent plan ownership.

Rebuilds compare exact mesh/options/ownership descriptors across ranks and require
monotone topology, geometry and partition generations. An unchanged generation
reuses all its plans. A replacement is constructed separately and published only
after every rank has succeeded. Host objects stay at fixed addresses while their
dependent operators exist. The previous generation becomes unusable as soon as
a rebuild starts, including when replacement fails. A local staging hook permits
the caller to prepare replacement field storage before publication; it does not
transfer or publish physical fields itself.

`Use`, sparse `Exchange`, and full `Gather` reject stale or incomplete generations
collectively. `Use` callbacks and the staging hook must contain only local work;
they synchronize exceptions before returning. Borrowed operators and stage views
must not survive a rebuild. Descriptor comparisons and donor searches occur at
rebuild events, not every RK stage. State-dependent derivatives and closure data
still require refresh at each changed stage.

The sparse exchange constructor now also synchronizes host/device allocation
exceptions before its next MPI operation or publication. This changes setup
failure handling, not the exchange arithmetic. Actual host/GPU out-of-memory
failures were not injected in this qualification.

## Local evidence

The Serial Kokkos backend passed with one and two MPI ranks:

| Test | Result | Wall time |
|---|---|---:|
| Complete bundle, one rank | 18 fresh/rebuilt comparisons; 24 builds | 105.98 s |
| Complete bundle, two ranks | 18 fresh/rebuilt comparisons per rank; 30 builds | 117.20 s |
| Existing sparse exchange, two ranks | All native checks passed | 0.89 s |

Each comparison uses freshly changed nonspherical analytic fields and actual
native projection, sampled-normal, angular, scri, bulk RHS and diagnostic
operators. Complete surface fields are gathered before bulk evaluation. Captured
operator arrays and diagnostic values match fresh independently constructed plans
byte for byte. The matrix covers Cartesian refinement/coarsening in the existing
mapped-grid fixture, changed shell geometry, repartitioning, and the continuous
spherical-overlap core. It also checks unchanged reuse, a single-rank invalidation
or stale key, changed descriptors without a new generation, inconsistent rank
descriptors, wrong/aliased gather storage, and one-rank staging failure/recovery.

Mixed refinement levels in the continuous Cartesian core remain unsupported and
are explicitly rejected. These tests do not evolve an AMR mesh, transfer fields,
test RK rollback/restart through a rebuild, or establish stability/convergence.
They are not GPU qualification.

Raw logs and receipts in [local/](local/) pin 50 source files before and after
every build/test command. Supplementary provenance records the unchanged fixture
and sparse-test sources, build cache and executables after qualification. The
reproduction script now includes those two fixture sources in its continuous
source checks and directly records the sparse-test executable hash as well.
The stored run records retain their original contents.

From the repository root, using the existing local MPI/Serial configuration:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/mesh-plan-bundle-v1/run_local.py \
  --output-dir output/hyperbolic-core-shell/mesh-plan-bundle-fresh
```

The output directory must be new. Builds use `-j2`; each command is bounded by
420 seconds. This local harness uses `/opt/homebrew/bin/mpiexec`. The CMake target
`hyperbolic_shift_3d_mesh_plan_bundle` and its registered two-rank CTest also
support the configured MPI launcher.

## Remaining integration

The planned evolution target now checks fixed-mesh CPU/MPI restart and rejected
stages against V24, including replacement of all plans on an unchanged layout.
Changing ownership or the physical mesh still requires replacement field storage
and a transfer policy before publishing a generation, with corresponding
diagnostic/checkpoint support. SYCL/MPI qualification and a controlled performance
comparison of that integration remain outstanding. Additional collective guards
must be measured in that comparison. These lifecycle checks do not add an AMR
evolution claim or change the existing lapse-floor policy.

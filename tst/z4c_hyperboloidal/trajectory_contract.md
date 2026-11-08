# Original-driver N12 trajectory comparison

`trajectory_oracle.py` builds a test entry point against the actual AthenaK
executable's object files. It uses ordinary dynamic input with
`integrator=rk4_classical`, `nlim=8`, `tlim=1`, and the pinned exact timestep.
The test calls the original `Driver::Initialize` and `Driver::Execute` once.
It does not overwrite the driver's integrator, timestep, or evolution mode.
All 106 native leaves retain equal load-balancing weights.

Observer tasks use the original native task lists. A before-stage observer
poisons unused physical ghosts, residual workspace, and RHS/status arrays.
After-stage observers verify all 32 stage clocks and record compatibility,
clock, and minimum chi. After-time-integrator observers, dependent on every
existing task in that list, capture the actual accepted states at steps 1 and
8. They neither project nor otherwise change the captured state. The fresh
problem-generator state is also saved before initialization.

The independent reference is the preserved CPU `SphericalTransfer3D::Step3D`.
Its classical RK4 numerator `k1+2*k2+2*k3+k4` and final `/6` match the original
GPU spherical-policy branch, which clears compensation. The GPU branch also
applies the selected chi policy after each intermediate and accepted candidate.
The runner creates a private copy of the CPU header with hooks at precisely
those candidate locations and at the end of each RHS evaluation. The hook uses
the read-only original `EndpointChiCandidate` implementation and records all
eight ledger members. The script proves that reversing its declared hook
insertions recovers the original header byte for byte. Original, overlay, floor
helper, and GPU driver source hashes are recorded.

Both trajectories must have zero floor events through step 8. Thus adding the
campaign's floor policy to the CPU reference cannot change this short
trajectory's values. Native and reference state, per-cell minima and accepted
minima, event counts, nonpositive/nonfinite counts, corrections, and first-event
times are compared. Every true RHS stage's compatibility must remain at most
`1e-7`; compatibility, clock, chi minima and exact stage times are also compared.
The native initialization's additional compatibility solve is not counted as an
RK stage.

The reference mesh is constructed independently. Native and reference leaf
bounds and every bulk sample coordinate must agree exactly. Values are matched
by native leaf GID and logical sample indices; all 31 fields at 229392 bulk and
864 surface points are included. Initial states must be bitwise equal. The
previously adopted short-trajectory gate remains
`abs(delta)<=1e-12+5e-10*abs(reference)`, with chi's absolute term tightened to
`1e-14`. Ledger minima use that same tighter chi gate; other ledger members are
required to agree exactly.

For each snapshot, machine JSON includes per-field maximum absolute/scaled
errors, physical-volume-weighted RMS, and worst absolute/scaled locations with
coordinates and values. Bulk RMS includes every independently evolved overlap
point using the reference physical cell volume and outer endpoint-grid half
weight. Surface RMS uses area weights. Complete native/reference bulk and
surface binary snapshots and native chi ledgers are written in canonical native
GID order, enabling independent bytewise comparisons between 2 and 8 local MPI
ranks. Binary file SHA-256 hashes are included.

Use `--build-only` to compile without starting a run, then `--run-existing` with
the same output directory. The latter requires matching source, input, native
executable, oracle executable, and legacy reference/header hashes. Reference
provenance is checked before rewriting the private overlay. `--run-output`
places logs and snapshots in a different directory while using the exact same
oracle executable and input, enabling a controlled decomposition comparison.
The runner checks source stability through the run. These are bounded local fixed-topology tests, not Aurora GPU,
AMR migration, restart, long-trajectory, or performance results.

The archived local results in `build-native/trajectory-results.json` used one
executable on both 2 and 8 ranks. The complete accepted bulk and surface files
at steps 1 and 8 agree byte for byte with the independent reference and between
decompositions. All per-field errors and weighted RMS values are zero; every
ChiRecord member and all 32 stage compatibility/clock/chi values agree exactly.
Postchecks verified 26784 surface values and 864 distinct complete field rows
per saved surface snapshot, plus all 864 eight-member surface ChiRecords.

After those runs, review prompted explicit global surface-target uniqueness
and legacy/native leaf-bijection preflights, and the legacy provenance reuse
guard described above. These additional checks were not executed in the
archived trajectory runs. Their original tested C++ source is preserved in
`build-native/trajectory-oracle-two/archived-test-source`; the next freshly
built oracle will execute the strengthened preflights.

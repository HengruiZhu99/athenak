# Optional V7 inner-transition interpolation: native implementation

`hyperbolic_shift_3d_mpi_inner_transition` selects the targeted logical-coordinate
halo interpolation diagnosed in the [V6 direction study](../scri-v6-interface-diagnosis-v1/README.md).
The physical-radius chains remain at spherical radial seams and in the outer
layers. V6's seven-node normal derivative and last-three-row scri operator,
independent Lambda/Theta, dependent-D/null reconstruction, damping and angular
filter are retained. No lapse floor, state reset or constraint overwrite is
introduced. Production hyperboloidal evolution remains disabled.

The selection is applied to `model.grid` before constructing device exchanges;
both volume-owned and surface-donor-owned halo exchanges use that grid. The
preconstructed scri sampling/radial plans remain unchanged. Host diagnostics
that use the grid's interpolation, including horizon geometry sampling, also
see the selected interpolation. No continuum equation is changed.

V7 uses magic `HYP3D_MPI_V7` with the exact104-byte header and normal tuple
**(3,7,4,3)**. V6 retains **(2,7,4,3)**. Cross-schema, wrong-tuple, unadmitted,
truncated and altered-epsilon restarts are rejected. Reports must state
`interpatch_interpolation=logical_inner_transition`. The checkpoint exporter,
Aurora controller/preflight and offline reanalysis require that exact policy;
reanalysis selects it explicitly rather than treating V7 as V6.

Local Serial-backend component checks establish complete CPU/device halo,
bulk-RHS, numerator, scri-RHS and dependent-D agreement for equilibrium and
nonspherical fixtures, with **zero differences**. Poisoned exterior radial
ghosts remain unused. Twelve simulated ownership partitions agree; this is
not an actual twelve-rank GPU test.

Actual one/two-rank MPI four-step pulse tests produce **bitwise identical full
checkpoints**, including both RK compensation arrays. A two-plus-two restart
matches uninterrupted evolution. The rebuilt V6 control matches the preserved
V6 executable bitwise. Short Minkowski and epsilon-zero controls pass.
The Python controller, preflight and exporter suite passes **86 tests**,
including rejection of missing V7 interpolation identity and full-exchange
component evidence.

These are implementation/restart tests only. There is **no V7 Aurora duration,
100 M, puncture-convergence or AMR acceptance** here. The diagnosis suggests a
candidate, not a global energy estimate. The previously observed V6 negative
puncture lapse still requires independent resolution/timestep investigation.

Small logs, MPI verification and source hashes are retained here. Full local
checkpoints and binaries remain in
`output/hyperbolic-core-shell/inner-transition-native-local-v1`.
Build the `hyperbolic_shift_3d_inner_transition_device` and
`hyperbolic_shift_3d_mpi_inner_transition` targets. The complete MPI commands
and binary/artifact hashes are recorded in `mpi-verification.json`.

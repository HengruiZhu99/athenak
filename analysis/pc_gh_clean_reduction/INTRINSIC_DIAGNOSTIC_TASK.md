# In-process intrinsic component diagnostics

Enable with `pc_gh/intrinsic_diagnostics=true`. The default is false.
`pc_gh/intrinsic_diagnostic_dcycle` defaults to 1 and must be positive.
The existing synchronized constraint task now invokes the intrinsic diagnostic
after initialization (including restart initialization) and at the final RK3
stage on selected cycles. Initial observations are always written. Intermediate
stage data remain available through the separate stage-dump instrument.

The task first materializes primary `(w,rho,K,g,A)`, physical Q=J(s)S, and the
ten configuration potentials on the already valid state ghost cells. The
primary physical kernel reads only the first group. Reductions differentiate
the materialized potentials; in particular alpha=rho*w is multiplied at every
stencil point. Intrinsic curls use the original independent auxiliaries and
raw Q curls use the materialized Q. All first/second/mixed derivatives use the
shared finite-difference implementation, with three-cell per-axis reach at FD6.
No derivative of an unsynchronized derived field or RHS ghost is taken.

The 89 individually named components are H (1), M (3), alpha-M (3), C (1),
Z (3), reductions (30), intrinsic curls (30), and symmetric raw Q curls (18).
The physical operator and analytic convergence evidence are described in
`INTRINSIC_PHYSICAL_STENCIL.md`. This task does not contract the evolution's
Rstar to stand in for independent physical H.

For every component the task computes coordinate-volume L1 integral, L2
integral, RMS, absolute maximum and its signed value. MPI gathers local
contributions and selects a global winner, breaking exact ties by the smallest
global block ID, then the first local k/j/i. The output includes actual region
volume and cell count, global block ID, logical level, active-cell local k/j/i,
and physical cell-center coordinates. Each cell has one owner; ghosts never
contribute to norms. In 2D, coordinate volume includes the inactive dimension's
declared width, consistent with the existing independent global diagnostic.

Rank zero writes `intrinsic-diagnostics-c<CYCLE>-s<STAGE>.csv`. The time/cycle
refer to the synchronized state: initialization uses current restart time;
the final-stage task adds the pending step's dt/cycle increment. These times
were checked against actual float64 restart metadata. Stage zero identifies
initialization, including restart initialization. Every file is created
exclusively; an existing name is an error, not a silently overwritten or
appended rollback epoch. Continue overlapping epochs in a new run directory.
This new CSV format is separate from unsupported legacy `hst` output.

## Executed controls

Evidence: `qualification-runs-20260907/pcgh-clean-reduction/intrinsic-diagnostic-task-001/`.
Source and exact serial/MPI binaries are identified by manifests; raw restarts
remain in the external run root recorded there.

Six FD2/4/6 2D/3D cases each run serial and with two verified MPI ranks, at
initial and final synchronized states. They use independent oblique modes in
all 50 fields. All 89 component norms, signed values at the reported maxima,
coordinates, levels, volumes and cell counts are checked against independent
global-array diagnostics. Physical H/M use an independently assembled direct
primary metric-jet operator; historical repeated-derivative diagnostics remain
the default and are regression tested separately.

The maximum normalized component discrepancy is 1.142e-13 against the fixed
2e-12 tolerance. Coordinate discrepancies are at most 2.221e-16. Enabling the
diagnostic leaves the final float64 state bitwise unchanged in all twelve runs.
Four additional controls verify invalid-cadence rejection, cadence two over a
three-step run, restart continuation with bitwise identical history/state, and
preservation of an existing output file on a name collision. All nineteen legacy
restart/layout controls pass after the new task hook.

```sh
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_diagnostic_task.py \
  --binary BUILD/src/athena --fixtures SEEDED_DECOMPOSITION_RUN --output NEW_RUN
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_diagnostic_task.py \
  --binary MPI_BUILD/src/athena --fixtures SEEDED_DECOMPOSITION_RUN \
  --launcher '/opt/homebrew/bin/mpiexec -n 2' --ranks 2 --output NEW_MPI_RUN
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_diagnostic_controls.py \
  --binary BUILD/src/athena --fixtures SEEDED_DECOMPOSITION_RUN --output NEW_CONTROLS
```

## Scope still open

This checkpoint enables actual in-process full-region diagnostics on the
currently admitted uniform periodic meshes. No excised region is introduced or
claimed, and intrinsic nonconforming/physical-boundary paths are still rejected.
CUDA verification of this new task is not run. It allocates temporary geometry
and diagnostic arrays and gathers norms on the host; performance at production
scale is not qualified. These numerical operator checks do not establish smooth
physical evolution convergence, puncture regularity, or binary qualification.
Next: verify the new task on CUDA, then use its histories in the smooth/forced
evolution ladders while implementing the remaining interface requirements.

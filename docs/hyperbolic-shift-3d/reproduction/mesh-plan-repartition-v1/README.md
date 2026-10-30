# Live repartitioning of the planned driver

The opt-in planned evolution target can now change contiguous volume ownership
between accepted steps. It gathers the old owners' physical fields and RK
compensation, stages replacement local storage and all four volume RK rate
buffers, and builds a complete new mesh-plan generation. Publication happens only
after every rank has constructed the replacement. New halo plans bind the current
surface storage; reference lapse and angular scratch handles are refreshed;
boundary-stage and padded diagnostic caches are invalidated. Sparse donor
exchange then fills the new dependencies. Cell ordering, equations and RK
arithmetic do not change.

The surface partition stays fixed, so its fields, compensation and stage buffers
keep their existing storage. Floor history remains cumulative across the change.
If constructing the replacement fails, accepted fields and compensation have not
been published. The caller explicitly rebuilds the previous layout before
rethrowing the error, allowing checkpoint/failure capture to use a valid plan.
The native fixture tests an invalid descriptor on just one rank; it does not
inject actual device out-of-memory failures.

`HYPERBOLIC_PLAN_REPARTITION_INTERVAL=1` is a deterministic stress policy: after
each accepted nonzero step it alternates equal cuts and compressed cuts. With two
ranks these are 50/50 and 10/90 percent ownership, moving complete Cartesian
blocks as well as shell cells. This is not a measured-cost balancing policy.
The `Repartition(cuts)` API accepts independently selected contiguous cuts.
Default runs leave this setting at zero. Fixed-layout rebuilds preserve the
current cuts after repartitioning. Reports record actual cuts, change times,
generation counts and rejected transactions. Sparse operator counters explicitly
describe the current generation; end-to-end phase timers remain cumulative.

## Local evidence

All nine commands in [local/verification.json](local/verification.json) passed
with Serial Kokkos and one/two MPI ranks:

- The native transaction fixture preserves complete accepted fields and nonzero
  RK compensation byte for byte through a one-rank descriptor rejection,
  recovery, repartition and subsequent fixed-layout rebuild. A checkpoint is
  written after recovery. Continued evolution, rollback and complete failed-stage
  capture match an independent unrepartitioned evolution exactly.
- Pulse checkpoints match the separately frozen V24 control through two ownership
  changes, including a checkpoint written under unequal ownership and resumed
  with a different initial partition.
- Minkowski and isotropic puncture checkpoints match the frozen control exactly.
  Puncture diagnostics run at every step; complete accepted horizons are retained
  when Cartesian blocks change rank. Final horizon and exterior constraints agree
  with the frozen control at matching diagnostic cadence within 2e-12 relative
  tolerance (1e-14 absolute).
- Combined ownership changes and forced fixed-layout rebuilds preserve the same
  complete pulse checkpoint.

All evolved cases have zero lapse-floor interventions and surface/ghost null
residuals below 5e-13. These short tests do not establish long-time stability,
convergence or a new performance result. The frozen control executable is still
SHA256 `e4137758a1315a2e6b9fc033fc8d1bb2e9ac32de948c3a98db7848acd6e43353`.
The runner pins source, executables and initial data throughout execution and
compares complete checkpoint files. Small raw logs, histories and reports are
retained here; large checkpoints stay in local output.

An investigated horizon-ownership concern was ruled out: nominal `n^3` cells for
all blocks form the active-array prefix, with evolved overlap cells appended.
The existing central nominal-cell ownership rule therefore remains valid. No
horizon formula or sampling rule was changed for this work.

Build `hyperbolic_shift_3d_mpi_continuous_positive_planned` and
`hyperbolic_shift_3d_repartition_transaction` with `cmake --build
build-hyperbolic-mpi -j2 --target ...`. Reproduce with `run_local.py`; its required
options are `--binary`, `--transaction`, `--control`, `--initial`, `--mpiexec`,
`--reference-dir` and a fresh `--output-dir`. The retained verification record
contains exact executable commands. The transaction fixture is bounded by 420
seconds; each CLI evolution is bounded by 150 seconds.

The Aurora controller now recognizes the planned target as a distinct execution
identity. Normal stability admission requires coordinated generation ownership,
field batching, sparse exchange, and no forced rebuild/repartition settings.
Missing, mistyped or incompatible identities are rejected. All 106 offline
controller/worker tests passed. This prepares GPU qualification; it does not issue
GPU admission for the planned target.

## Outstanding scope

SYCL qualification and controlled integrated-driver timing remain outstanding.
Changing mesh geometry or topology still needs a physical field-transfer policy
and a corresponding checkpoint schema. The independent bundle tests construct
plans on refined/coarsened fixtures, but live AMR evolution is not supported by
this repartition change. Mixed refinement levels in the continuous-core stencil
remain unsupported.

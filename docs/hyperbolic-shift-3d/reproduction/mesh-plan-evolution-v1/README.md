# Coordinated plans in the native evolution driver

The opt-in `hyperbolic_shift_3d_mpi_continuous_positive_planned` target adopts the
complete [host/device plan owner](../mesh-plan-bundle-v1/README.md). Its fixed-mesh
trajectory matches the separately frozen, previously qualified V24 executable.
This is lifecycle integration and fidelity evidence, not a new performance result.

The owner constructs the host grid once, derives equal-cell ownership, and builds
the bulk, halo, scri, sampled-normal, angular and sparse-MPI plans as one generation.
All operator lookups check that generation locally; stage boundaries also check
it collectively. Surface and volume gathers use the owner's cached counts and
displacements. Initial field data and accumulated diagnostic maxima now belong
to the evolution driver, so they do not require modifying the owner's host model.

The default run builds once and reuses the generation through all RK stages,
diagnostics and checkpoint writes. Geometry-dependent reference lapse is reused;
state-dependent derivatives, normal samples and closure quantities are recomputed
according to the existing stage policy. Physical equations, interpolation and
derivative sums, log chi/lapse integration, lapse floor, algebraic projection,
null closure and V11 checkpoint bytes are unchanged.

## Accepted-step replacement

`RebuildCurrentPlans` replaces the complete generation at a synchronization point
on the **same physical mesh and ownership**. The replacement is built separately,
its layout is checked against existing fields, and both halo operators are bound
to the current surface storage before publication. The driver then replaces its
reference-lapse/angular-buffer handles, clears its padded diagnostic snapshot,
and invalidates its boundary-stage cache. Physical fields, RK compensation and
floor history retain their existing storage and values. Operator pointers and
host-model references are fetched anew, including by the horizon sampler.

`HYPERBOLIC_PLAN_REBUILD_INTERVAL=1` exercises this path before every step. Rank
zero validates the setting and broadcasts it. The default is zero. Reports expose
the interval, generation/build counts and rank-zero rebuild time. Setup timing
uses slot 6 for complete bundle construction and slot 7 for field storage; the
`mesh_plan_setup_layout` field identifies this distinction from older targets.
Rebuild time is separate from the reported step timer. Forced-rebuild tests are
not performance benchmarks.

## Verified local results

The original eleven-case poisoned-workspace matrix passed in 148.18 seconds with
the Serial Kokkos backend. All candidate cases retained exactly one generation.
Complete checkpoints match the frozen control byte for byte for the pulse with
one/two MPI ranks and a one-to-two-rank restart, and for Minkowski and isotropic
puncture with two ranks. The deliberately rejected RK stage preserves the same
rollback checkpoint and complete failed-stage capture.

Six further two-rank comparisons passed:

| Case | Rebuilt generations | Exact comparison |
|---|---:|---|
| Refactored legacy execution | 1, independent plans | Pulse checkpoint |
| Pulse, rebuild before every step | 4 | Three-step checkpoint |
| Pulse restart with rebuilds | 3 | Uninterrupted control checkpoint |
| Minkowski with rebuilds | 3 | Two-step checkpoint |
| Isotropic puncture with rebuilds | 3 | Two-step checkpoint |
| Rejected RK stage after rebuild | 2 | Rollback and failed-stage bytes |

The largest recorded surface null residual in these candidate cases is about
8.89e-16. Every case has zero lapse-floor node interventions. These are short
fidelity checks; they do not establish longer-time stability or convergence.

The native bundle matrix was also rerun with the updated owner and passed on
one/two ranks in 105.96/122.32 seconds. It now compares automatic equal-cell cuts
against explicit cuts and directly rejects borrowed access before initialization,
with a wrong epoch, and after a failed rebuild. Its 18 native fresh/rebuilt
comparisons per rank still cover refinement/coarsening in the mapped-grid fixture,
changed shell geometry, repartitioning, and the continuous-core configuration.
The existing two-rank sparse-exchange unit test passed as well. These raw checks
are retained in `local/bundle/`.

The frozen control SHA256 is
`e4137758a1315a2e6b9fc033fc8d1bb2e9ac32de948c3a98db7848acd6e43353`,
matching the previously retained [V24 binary record](../radial-field-batch-v1/local/binaries.json).
It was copied before rebuilding either target. `run_parity.py` and
`run_rebuilds.py` retain commands, source/executable hashes and complete-file
comparison hashes. Small raw reports, histories and logs are in `local/`;
large checkpoints and rejected-stage captures remain in local output.

Build with the existing local MPI configuration:

```sh
cmake -S . -B build-hyperbolic-mpi
cmake --build build-hyperbolic-mpi -j2 --target \
  hyperbolic_shift_3d_mpi_continuous_positive_planned \
  hyperbolic_shift_3d_mpi_continuous_positive_radial_batch
```

Run `run_parity.py` with `--binary`, `--control`, `--initial`, `--mpiexec` and a
fresh `--output-dir`. The control should be the separately preserved executable
with the hash above. Then run `run_rebuilds.py` with `--binary`, `--legacy`,
`--initial`, `--mpiexec`, `--parity-dir` pointing to that completed run and a fresh
`--output-dir`. The retained run receipts give exact commands and input paths.
The parity matrix has a 410-second internal/420-second wrapper deadline; each
additional rebuild case is bounded by 150 seconds.

## Remaining work

Live repartitioning requires transfer of owned fields and RK compensation before
publishing a changed layout. Geometry/topology changes additionally require a
field-transfer policy and a checkpoint schema describing the changed mesh. This
driver entry point cannot yet perform those operations. The independent bundle
fixture covers construction/rejection after refinement, coarsening, geometry and
ownership changes, but that is not AMR evolution support; the continuous-core
mixed-level stencil remains unsupported.

SYCL qualification, GPU rejected-stage capture and a controlled timing comparison
of the integrated driver remain outstanding. Collective generation checks add
communication that must be measured. The earlier 1.92x V24 batching result does
not measure this integration, and no new speedup is claimed here. Existing
late-time horizon-quality and convergence limitations remain unchanged.

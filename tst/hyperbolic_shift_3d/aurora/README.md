# Aurora debug deployment

These files run the new hyperboloidal Cartesian-core/shell research scheme. The coordinator is the sole submitter. They do not run standard Z4c regression campaigns or claim production AMR support.

The user-authorized policy is **debug, one node, all 12 MPI ranks, one GPU tile per rank, one allocation at a time**. `control.py` rejects other resources. Debug currently permits walltimes from 5 minutes through 1 hour. The worker leaves at least 60 seconds before the scheduler deadline, and each evolution receives a shorter numeric checkpoint deadline. A `checkpointed` result is resumable evidence, never a completed stability pass.

## Connection and fresh directories

Verified 2026-10-28 from the coordinator's host:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=15 \
  -o ControlPath=/Users/hz0693/.ssh/codex-control/%C \
  hzhu@aurora.alcf.anl.gov
```

All new campaign files must remain in home, with no more than 5,000,000,000 additional bytes. Use fresh remote root `/home/hzhu/hyperbolic-core-shell-20261028`, with `source/`, `build-sycl-mpi/`, `inputs/`, and `campaigns/`. Preserve old frozen exports. Transfer the full current source, including newly added launcher and evolution files; write a manifest mapping root-relative source filenames to SHA256 hashes. Record Git revision plus uncommitted diff separately if source is not yet committed. Freeze this tree before binding a plan.

For Kokkos, the read-only export `/lus/flare/projects/CompactBinaryMerger/hzhu/cubed-sphere-20261012/source/kokkos` matches all **1,486** tracked files at local pin `6739bc623081648af9e752b616d9671527922cbf`. SHA256 of the canonical JSON filename-to-content-hash mapping (sorted keys, separators `,` and `:`) is `c190efae9a0b15fe2f3d2b3f91e415f7779fada351f66311ba81009867b25ce0`. The dependency has been copied and verified in the new home source tree; the temporary campaign scratch copy was removed after verification. Do not modify the old frozen export. It has no Git metadata. Include copied Kokkos files in the new source manifest.

## New MPI + SYCL build

From the fresh remote root, after the MPI target is present:

```sh
module load oneapi/release/2025.3.1
cmake -S source -B build-sycl-mpi \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int' \
  -DCMAKE_CXX_FLAGS_RELEASE=-O2 \
  -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
  -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON \
  -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_INTEL_PVC=ON
cmake --build build-sycl-mpi --target hyperbolic_shift_3d_mpi -j 64
```

Build the actual parity/restart test targets as well. Retain configure/build logs, `CMakeCache.txt`, `module list`, `icpx --version`, `mpicxx -show`, and executable hashes. The named module currently resolves compiler 2025.3.2; record the actual version, not just the module label. Historical binaries were MPI-disabled and must not be reused. `/usr/bin/python3.11` needs no third-party packages for these control files; it currently lacks NumPy, so prepare numerical initial data separately and bind its hash.

## GPU placement

The PBS template selects `OMP_NUM_THREADS=1`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, and `MPIR_CVAR_ENABLE_GPU=1`. Each command launches:

```sh
mpiexec -n 12 -ppn 12 \
  --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 \
  gpu_tile_compact.sh EXECUTABLE ARGUMENTS
```

The installed ALCF wrapper maps local rank `r` to `ZE_AFFINITY_MASK=(r/2).(r%2)`, giving the 12 distinct tiles of the six GPUs. Its installed path at verification was `/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh`. A 12-rank affinity probe must pass before scientific commands run. Placement alone is not evidence of useful GPU execution; the numerical report must verify backend, rank count, CPU/GPU parity, and distributed work. Guidance: [ALCF Aurora job launching](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/).

## Reviewed command plans and evidence

Copy `plan.example.json` into a fresh `campaigns/<name>/plan.json`. It is deliberately unreviewed and incomplete. Replace/extend it with the actual parity and restart cases and numerical acceptance criteria. Every command is an explicit argument list, with no shell interpolation. `@WALL_SECONDS@` is the one supported argument substitution and becomes a numeric deadline with 30 seconds left for shutdown. The main evolution interface is:

```text
n output_prefix kind target_time dt initial_file|- restart_checkpoint|- wall_seconds [diagnostic_interval [horizon_rms_tolerance [horizon_lmax [horizon_ntheta]]]]
```

The optional targets `hyperbolic_shift_3d_mpi_dissipative` (V4) and
`hyperbolic_shift_3d_mpi_sampled_normal` (V5) accept a
13th positional argument, `scri_dissipation_epsilon`, after all twelve arguments
above. Its default is 2 when omitted. Epsilon must be finite and nonnegative;
epsilon zero retains the V4 schema and degree 7. All older targets reject a
13th argument. The native report records `scri_dissipation_epsilon`,
`scri_dissipation_degree` and `checkpoint_header_bytes`; add exact checks for
these fields to every reviewed V4 evolution plan. The original V1
resolution-control failure-continuation policy is not available to V4.
It is also unavailable to V5. V5 has no normal-closure CLI switch: its target
selects the sampled-native-numerator scheme, independently of epsilon. Its
reports must explicitly state `scri_normal_derivative=sampled_native_numerator6`,
`scri_normal_radial_nodes=6`, `scri_normal_bulk_derivative_order=4` and
`checkpoint_header_bytes=104`. The optional CPU manufactured study observes
roughly third-order complete surface-rate errors on its tested fixtures;
neither that result nor a favorable projected directional rate is a stability
or trajectory-convergence admission.

`kind` is `minkowski`, `pulse`, or `puncture`. The puncture donor mass is `M=1/3`; 100 M means target coordinate time `100/3`, not 100. Every command needs a JSON report with `status=completed|checkpointed|failed`. Checks use explicit JSON paths and `eq`, `le`, `ge`, or `finite`; configure checks against the final executable schema. `inputs` are existing initial/checkpoint files bound before submission. `artifacts` are output files hashed after execution. A later case may name earlier artifacts as `generated_inputs`; the worker records their hashes immediately before use. Evolution argument grid, kind, output prefix, initial/restart declarations, and bounded deadline are checked against plan metadata. Source, executables, inputs, command plan, launcher files, PBS script, and admission evidence are bound and reverified.

From the remote root, these are distinct operations:

```sh
python3 source/tst/hyperbolic_shift_3d/aurora/control.py bind \
  --root "$PWD" --plan campaigns/preflight-v1/plan.json --source-manifest source-manifest.json
python3 source/tst/hyperbolic_shift_3d/aurora/control.py verify \
  --root "$PWD" --plan campaigns/preflight-v1/plan.json
# Only the coordinating agent invokes this explicit submission command:
python3 source/tst/hyperbolic_shift_3d/aurora/control.py submit \
  --root "$PWD" --plan campaigns/preflight-v1/plan.json
python3 source/tst/hyperbolic_shift_3d/aurora/control.py collect \
  --root "$PWD" --plan campaigns/preflight-v1/plan.json
```

`submit` queries the exact campaign job name `hyper-scri-3d` and blocks queued/running/held task jobs. Other held research jobs are untouched. It then atomically acquires a root-wide allocation lock and a permanent per-plan submission lock **before** calling `qsub`. Nonzero exits, connection loss, or an ambiguous scheduler ID retain the locks and pending record. Do not retry submission; inspect the saved output and scheduler to establish the original result. A fresh plan cannot bypass the root allocation lock. Once `qstat` explicitly reports that the job is terminal, collect with `--release-terminal-lock` to release only the root lock. The original plan remains non-resubmittable. Never delete uncertain locks merely to continue.

Reports, stdout, hashes, and `worker-status.json` distinguish passed, checkpointed, failed, external timeout, and incomplete runs. A short pass cannot admit a long stability claim. A checkpointed worker stops its command plan; resume via a fresh plan that binds the accepted checkpoint and the same scientific parameters. Allocation completion and scientific completion are separate records.

1. **Preflight:** fixed low grid (initially `n=16`), GPU/CPU parity, distributed rank coverage, restart equivalence, then short Minkowski equilibrium/pulse and puncture checks.
2. **Stability:** same grid, extend Minkowski/pulse and puncture evolution to the requested duration before increasing resolution. Diagnose failures and repeat preflight when code changes. Tune timestep and runtime using measurements.
3. **Convergence:** only after all low-grid stability tests pass, including puncture to 100 M; then refine spatial resolution and independently refine timestep.

Phase transitions require a coordinator-reviewed admission JSON. Preflight admission needs `passed=true`, `phase=preflight`, `low_resolution_n`, `source_manifest_sha256`, executable `binaries` hash mapping, `gpu_cpu_parity=true`, and `restart_equivalence=true`. Stability admission has `phase=stability`, the same bindings, `minkowski_stable=true`, `pulse_stable=true`, `puncture_stable=true`, and `puncture_completed_t_over_M>=100`. Retain the exact supporting reports/checkpoint hashes and numerical thresholds in these evidence documents. The controller checks the admission fields and hashes; the coordinator must assess the actual scientific evidence.

V4 admissions additionally require `scri_dissipation_configs`, a list of exact
configuration records. For example, a separately verified epsilon-2 record is:

```json
{
  "executable": "build-sycl-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_dissipative",
  "checkpoint_schema": "HYP3D_MPI_V4",
  "scri_dissipation_epsilon": 2.0,
  "scri_dissipation_degree": 7,
  "checkpoint_header_bytes": 88
}
```

Each requested epsilon must have its own supported record and exact executable
path. A scan of 0, 1 and 2 therefore needs all three configurations admitted;
epsilon 2 does not authorize 0 or 1 even when source and executable hashes
match. Omitting the runtime epsilon requests the default 2 and still requires
that record. Existing V1–V3 admissions remain unchanged. Admissions are written
only after reviewing the corresponding source-bound reports and restart
comparisons; this list is not proof by itself.

V5 instead requires its own `normal_derivative_configs` list. A V4
`scri_dissipation_configs` record cannot qualify V5, even if epsilon is equal.
Each V5 record binds all of the following, in addition to the usual exact
source/binary/preflight evidence:

```json
{
  "executable": "build-sycl-mpi-v12/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_sampled_normal",
  "checkpoint_schema": "HYP3D_MPI_V5",
  "scri_dissipation_epsilon": 2.0,
  "scri_dissipation_degree": 7,
  "checkpoint_header_bytes": 104,
  "scri_normal_derivative": "sampled_native_numerator6",
  "scri_normal_radial_nodes": 6,
  "scri_normal_bulk_derivative_order": 4
}
```

This is a schema example, not an admission. There is no implicit reuse of a
failed V4 duration checkpoint. A new V5 duration begins from fresh analytic
initial data after its source-bound GPU parity and exact-restart preflight.

Offline launcher tests (no SSH or scheduler calls):

```sh
python3 tst/hyperbolic_shift_3d/aurora/test_control.py
```

## Home footprint and retention

Every reviewed plan must include `storage.limit_bytes=5000000000`, `storage.planned_new_bytes` (the sum of newly expected outputs/logs), and `storage.atomic_checkpoint_reserve_bytes` (at least one extra full checkpoint). These estimates are additional to the measured source/build/campaign footprint, not substitutes for it. At `n=16`, the current checkpoint is about 39 MB; use the actual measured size plus margin when setting the reserve. The example's larger estimates are placeholders.

`storage.py` counts the campaign tree conservatively using the larger of logical and allocated bytes, without following external symlinks. Binding/submission/start require space for all planned outputs and the reserve. During a command, the worker checks every five seconds that the checkpoint reserve remains available and terminates the command if the cap is threatened. All worker output must be under the home campaign root. The source/build stage must also stay within this same cap; build logs and generated object files count.

The scripts never delete files. The coordinator should regularly record checkpoint/failure-file hashes and comparison results, then delete only explicitly identified bulky artifacts that are no longer needed. Preserve the newest accepted checkpoint for each continuing evolution and the relevant failure evidence until analyzed. Keep reports, histories, small logs, parameter manifests, and hashes. Do not purge bound inputs for a pending/running plan or files in unrelated campaigns. Recheck the footprint before the next submission.

## Component and restart adapters

`preflight.py` runs under the worker's existing 12-rank launch. Each component rank executes its bound standalone C++ test once, logs separately, and sends its result over a local Unix socket. There is no nested `mpiexec`, mpi4py dependency, or filesystem-polling barrier. All 12 ranks must acknowledge the final result. This duplicates small component audits across GPU tiles; the evolution executable partitions the bulk work.

Common arguments come before the mode:

```text
source/tst/hyperbolic_shift_3d/aurora/preflight.py
  --root ROOT --plan campaigns/NAME/plan.json
  --output-prefix campaigns/NAME/output/parity --seconds @WALL_SECONDS@
  component --component device|exchange|scri
  --test-binary build-sycl-mpi/tst/hyperbolic_shift_3d/TEST
  --cmake-cache build-sycl-mpi/CMakeCache.txt
```

Use targets `hyperbolic_shift_3d_composite_device`, `hyperbolic_shift_3d_composite_exchange`, and `hyperbolic_shift_3d_scri_device`. Declare the binary and cache as bound `inputs`. The adapter requires actual `backend=SYCL` stdout on **every rank**, successful native assertions and marker, finite error metrics, and correct tile placement. Scri audits also require the dependent-input/order-independence checks. Artificial scri perturbation pole/clock values are parity data, not equilibrium tolerances. The aggregate report includes `status`, `mpi_size=12`, `backend`, `gpu_cpu_parity`, per-rank records and maximum metrics.

For V4 preflight, two additional component modes consume their matching
standalone targets:

- `--component angular_dissipation` runs
  `hyperbolic_shift_3d_angular_dissipation`. The adapter checks n6/n8 degree-7
  and n6 degree-5 reference cases, finite reference/device/partition errors,
  energy and adjoint errors, exact constant annihilation, moment and
  conditioning bounds, coefficient bounds, and a positive spectral bound.
- `--component scri_dissipation_closure` runs
  `hyperbolic_shift_3d_scri_dissipation_closure`. The adapter checks synthetic
  and actual-plan additive rates, moving-normal dependent-D response,
  CPU/device/partition parity, epsilon-zero equivalence, equilibrium and
  perturbed null/compatibility residuals, and explicit invalid-input/CFL guards.

Both require the exact native success marker and actual SYCL backend on every
rank. These adapters do not add a communication-benchmark rerun requirement.

V5 adds `--component sampled_normal_device`, using the target
`hyperbolic_shift_3d_sampled_normal_device`. Each tile must report both n6,
epsilon-2 equilibrium and nonspherical cases. The adapter requires CPU/device
pole, rate and dependent-trace errors at most `2e-10`, twelve-partition error
at most `2e-13`, exactly zero finite-D dependence, null error at most `3e-14`,
and compatibility error at most `2e-12`. It also requires exact default-path
bitwise and rejection-check flags, actual `backend=SYCL`, successful process
exit, and the exact native success marker. This component is a bounded parity
check; a passed component report does not admit duration stability.

Restart comparison uses the same common arguments followed by:

```text
restart --reference REFERENCE.checkpoint --candidate RESUMED.checkpoint
  --reference-report REFERENCE.report.json --candidate-report RESUMED.report.json
  --atol 0 --rtol 0
```

Run it after uninterrupted and split/restarted evolution cases in the same bound plan, declaring all four artifacts as `generated_inputs`. Only rank zero reads the checkpoint payload; all ranks receive the result. It validates the native header, configuration/time/step agreement, finite full state and compensation arrays, matching completed SYCL reports, hashes, and elementwise differences. The report exposes `restart_equivalence`, `bitwise_payload_equal`, maximum errors, and header metadata. Current source and plan identity are checked against the worker binding. For comparisons across different plans, independently verify the original source/executable provenance before admitting results.

V1–V3 retain their original 72-byte little-endian `<16s4i3Q2d` header and array
layout. V4 uses magic `HYP3D_MPI_V4` padded to 16 bytes, the same base header,
then exactly 16 bytes `<dii>` containing epsilon, degree 7 and reserved zero.
Its four binary64 arrays begin at byte **88**. Both the preflight reader and
`checkpoint_export.py` reject truncated/trailing data, invalid configuration,
and nonfinite payloads. V4 report configuration must be explicit and agree
with the checkpoint. Restart comparison requires exact schema, epsilon,
degree, clock and step metadata even when nonzero payload tolerances are
requested. Equal V3 and epsilon-zero V4 arrays do not make their schemas
interchangeable. Export metadata retains the schema, configuration and actual
header size.

V5 uses magic `HYP3D_MPI_V5`, the same 72-byte base and 16-byte dissipation
record, then a 16-byte little-endian `<4i>` normal-operator record. Its fields
are exactly `(scheme=1, radial_nodes=6, bulk_derivative_order=4, reserved=0)`.
All four arrays begin at byte **104**. Unknown normal identifiers, widths,
orders or reserved values, truncated/trailing data and mismatched/missing
normal report fields are rejected. Payload tolerances never relax schema or
configuration equality. The exporter retains all three normal metadata keys;
V1–V4 checkpoint bytes and reader behavior remain unchanged.

Additional offline tests exercise checkpoint corruption, exact/tolerant comparison, all three metric adapters, 12-process success/failure propagation, and storage accounting:

```sh
python3 tst/hyperbolic_shift_3d/aurora/test_preflight.py
```

# Native Aurora validation plan

Status: **fresh native N12 through step356 (0.5014M) independently admitted**
(2026-11-08). Job9051541 completed normally. All initial/endpoint fields,
available chi aggregates, eight RMS histories, horizon quantities,1424stage
records and frozen-runtime/cadence checks pass. The historical3888floor events
and eight distinct cells agree. See evidence/native-n12-common-01-science.json.
The first comparator attempt stopped on initial null-aggregate reporting; that
failure is preserved and the report-only fix changed no numerical gate.
Segment356→711 job9051587 is queued after its actual submission preflight. The
full3551-step trajectory and performance/scaling remain unproven. Earlier
24/96-rank short, stage/poison and observed-main restart/CFL qualifications remain
valid, with all31 prerequisite report bindings checked onAurora.
All eleven historical common snapshots are now copied and bound to the exact
Aurora map, with complete SP1-field agreement and pinned step3551 hashes.

The chronological preparation and retained failure evidence below explain how
the frozen build was qualified; earlier pending-job statements describe their
recorded stage, not current authorization or current results. Current goal
requirements are tracked in requirements.json and status.txt.

The private staging completed
at 06:40:19 UTC; the native SYCL build completed successfully with `make -j64`.
Its original AthenaK executable SHA-256 is
`f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c`.
Source/compile-policy checks passed and 228 object/library hashes are preserved
in [evidence/aurora-native-build.json](evidence/aurora-native-build.json) and
`build-native/aurora-native-build`. Replay job **9051373** was
submitted at 06:40:56 UTC and has completed successfully (PBS exit 0): its maximum
full/fixed-outer norm difference is `8.6736173798840355e-19`, with one successful
complete FastFlow horizon. Production files remain unchanged. This establishes
reference replay reproducibility, not native GPU correctness or scaling.

The immutable snapshot contains 1973 source/input files, with base Git commit
`fce02a15f3ee533a34f2ce7c5046fc204859983e`, explicitly recorded dirty diagnostics
changes, and Kokkos commit `6739bc623081648af9e752b616d9671527922cbf`. Archive SHA-256:
`cba4c977043eaffc1f35b95e8557376f53fd8589810e0c2267a6db9c09bd5c12`.
The subsequent explicitly authorized test-only native replay and gated original
native eight-step smoke both completed in two-node/24-rank job **9051398** (PBS
finished, exit 0). All strict saved-state, full/horizon-exterior history, horizon,
read-only, and complete canonical point-coverage checks passed. Maximum strict
norm difference was `2.185751579730777e-16`. Aurora host/device fixed-cut membership
agreed (9864 outside, 432 exactly at .94), whereas the local CPU replay had 9874
outside and 430 exactly at .94. This distinguishes architectures; the pointwise
cause of the cross-platform difference remains a separate attribution audit.
Both actual initial/eight-step native checkpoints were downloaded and all 85
run artifacts verified against individual hashes. Their complete state comparison
subsequently passed, as did the96-rank controls and full stage comparisons below.
No3551-step native evolution has run yet. See
[evidence/aurora-native-saved-replay-two.json](evidence/aurora-native-saved-replay-two.json).

The eight-step correctness log has cycle intervals roughly 4.7–5.8 seconds
before final output, unexpectedly slow for N12. These are planning observations,
not a controlled benchmark. The compiler also reports SIMD32/256-register kernels
with spills around 137/140 in `SurfaceExchange::Solve` and 46 in `Evaluate`.
Preserve both observations for profiling after correctness; do not submit the
full 3551-step run into a one-hour allocation assuming fast performance.

Local staging evidence is `build-native/aurora-stage-20261108T063914Z`; its
`staging-origin.json` maps every file to its local absolute origin and hash.
The same origin/package manifests are retained in the remote private root.

## Isolation and observed queue state

Use the new private destination
`/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side`.
It did not exist at inspection time. Keep `source`, `build-sycl`, `reference`,
and distinct `runs/<case>` directories under it. Do not reuse production build
outputs, write into `scri-ko01-20261106` or `scri-resolution-20261107`, resume
N24, or launch N36. The new N12 evolution starts from analytic initial data.

The existing SSH control master works through
`/Users/hz0693/.ssh/codex-control/9a355d7bb9bdc04368c5de4b6d3474a2c1512802`:

```sh
ssh -S /Users/hz0693/.ssh/codex-control/9a355d7bb9bdc04368c5de4b6d3474a2c1512802 \
  -o BatchMode=yes hzhu@aurora.alcf.anl.gov 'qstat -u hzhu; qstat -Qf debug; qstat -Qf debug-scaling'
```

At **2026-11-08 06:28:26 UTC**, `qstat -u hzhu` listed three old held capacity
jobs (8883678, 8888791, 8888792), and no running or queued user campaigns.
Other users had active jobs. Do not release, alter, or delete those held jobs.
Recheck the live queue before every eventual submission; this inventory is not
future permission or a claim that the machine is idle.

| Queue | Relevant observed limits | Planned use |
| --- | --- | --- |
| `debug` | 1–2 nodes, 5 minutes–1 hour, one running and one queued job/user | 2 nodes / 24 GPU-tile ranks |
| `debug-scaling` | 1–256 nodes, 5 minutes–1 hour, one running and one queued job/user | 8 nodes / 96 GPU-tile ranks |

The task limit is **eight nodes**, even though the second queue permits more.
Use account `CompactBinaryMerger`, filesystems `home:flare`, and serialized
bounded jobs. Evidence with exact stdout and command arguments is in
[evidence/aurora-native-readonly-inventory.json](evidence/aurora-native-readonly-inventory.json).

## Compiler, device, and binding identity

`module load oneapi/release/2025.3.1` currently selects **Intel oneAPI DPC++/C++
2025.3.2 (2025.3.2.20260112)**, not a compiler reporting version 2025.3.1.
This agrees with the compiler embedded in the surviving reference diagnostic.
The inspected path is
`/opt/aurora/26.26.0/oneapi/compiler/latest/bin/icpx`.
Other loaded modules are GCC 14.3.0, MPICH
`prd/5.0.0.aurora_test.87e2045`, libfabric 2.3.1, and cray-pals/cray-libpals 1.8.0.
CMake is 3.28.3. Record actual versions and paths on the eventual build day.

The preserved resident build uses Kokkos **4.7.2**, SYCL and SERIAL enabled,
INTEL_PVC architecture, in-order queues, and no SYCL relocatable device code.
Preserve double precision, `-fp-model=precise`, `-O2`, `-ffp-contract=off`,
`-fno-sycl-id-queries-fit-in-int`, `-fsycl-targets=spir64_gen`, and backend device
`12.60.7`. Kokkos's INTEL_PVC CMake selection supplies the last two device flags.
The prepared script verifies those flags in the actual native compile commands.
Build the native repository through its CMake entry point with `Athena_ENABLE_MPI=ON`;
do not reuse the predecessor's objects, generated MPI-disabled `config.hpp`,
legacy standalone macros, or `libcurrent_spherical.so` for the native executable.
The native CMake C++17 convention is retained, with the exact argv recorded.

Use twelve ranks per node, one per PVC tile, and one OpenMP thread per rank:

```sh
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1
unset ZE_AFFINITY_MASK
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
```

The official wrapper is
`/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh`, SHA-256
`4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd`.
For local rank `l`, it sets `ZE_AFFINITY_MASK=(l/2).(l%2)` and
`EnableImplicitScaling=0`. The prepared launcher checks the wrapper hash and
records/checks all effective local-rank masks before evolution. It creates
NEO/SYCL caches only in that private run directory.

## Prepared files and native invocation

The local files under `tst/z4c_hyperboloidal/aurora/` are:

- `n12.athinput`: fresh native N12, with real history/FastFlow and restart outputs.
- `build-native.sh`: new-directory-only CMake build and `make -j64`, native MPI/
  precision/device flag checks, executable hash, and unchanged-source checks.
- `source_manifest.py`: hashes build sources, Kokkos, generated-config template,
  and native test/run inputs before/after the build and each run.
- `run-native.sh`: 2 or 8 nodes; exactly 1, 8, or 3551 fresh complete steps;
  unique private case directory; allocation/binding/provenance checks.
- `run-reference-replay.sh`, `replay-two.pbs`: the single authorized 15-minute,
  two-node, immutable saved-state diagnostic replay; no evolution.
- `two.pbs`, `eight.pbs`: prepared one-hour queue wrappers. Their default is the
  eight-step check. Neither submits another job or schedules continuation.

Source staging used the frozen current diagnostics snapshot. Stage the complete
native source **and the pinned Kokkos submodule contents** (a plain top-level
`git archive` omits those contents). Record the main and Kokkos revisions,
archive hash, and source manifest in `source/source-revision.json`, or retain
valid independent Git metadata for both repositories. The build script supports
both forms. The executed staging used the non-Git form and verified every remote
file against its local-origin hash.

After source staging is separately authorized, the prepared build command is:

```sh
TASK_ROOT=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
bash "$TASK_ROOT/source/tst/z4c_hyperboloidal/aurora/build-native.sh" \
  "$TASK_ROOT/source" "$TASK_ROOT/build-sycl"
```

Inside the appropriate allocation, the evolution invocation is the original
AthenaK executable and its original `main`/`Driver::Initialize`, `Execute`, and
`Finalize` lifecycle (`src/main.cpp`, `src/driver/driver.cpp`):

```sh
# RANKS=24 on two nodes, RANKS=96 on eight nodes.
mpiexec -n "$RANKS" -ppn 12 --cpu-bind=list:"$binding" gpu_tile_compact.sh \
  "$TASK_ROOT/build-sycl/src/athena" -i "$CASE/input.athinput" -d "$CASE" \
  -t 00:50:00 time/nlim=3551
```

There is no restart argument or alternate evolution driver. `run-native.sh`
wraps this with a 3420-second external timeout and requires normal cycle-limit
termination, all expected cycle records, and unchanged exact dt. A wall-time
stop or timeout is an incomplete test; it is not reported as a 5M result.
The one-hour resource request is a bound, not a prediction of runtime.
After all preceding gates pass and submission is authorized, use the prepared
PBS file with explicit variables, for example `NATIVE_STEPS=1` and a new
`NATIVE_CASE=n12-nodes2-step1`. Only after the short gates should
`NATIVE_STEPS=3551` be used. Reuse the **same executable** at both node counts.

## Exact N12 workload

The source of truth is [N12-reference.md](N12-reference.md) and the unchanged
[comparison-protocol.md](comparison-protocol.md). Preserve:

| Property | Value |
| --- | --- |
| Mesh | N12, 106 leaves; 34 Cartesian (central SMR), 72 shell leaves |
| Accepted values | 229392 bulk points and 864 scri points, all 31 fields plus chi records |
| Time integration | Native `rk4_classical`, 3551 complete steps |
| Nominal dt | `0.00046944203735617445` exactly |
| Final coordinate time | `1.6669886746517166`, corresponding to `5.0009660239551499 M` |
| Limit | `nlim=3551`, `tlim=2`, preventing clipping at exactly 5M |
| KO / stored chi floor | `.1` / `1e-10` |
| Gauge | Migrated constant shift-speed policy and pinned reference constants |
| Tracking | Disabled; fixed horizon center at `[0,0,0]` |
| FastFlow | lmax 12, ntheta 24, max 300 iterations, alpha-beta .1, initial radius .02, mass tolerance 1e-10, expansion RMS tolerance 1e-3 |

History and restart cadence is coordinate time `1/6` (0.5M); the original Driver
also writes final output. Validate the actual emitted common times/cycles against
the old list (0, 356, 711, 1066, 1421, 1776, 2131, …, 3551), rather than assuming
floating-point output scheduling selected identical steps. The prepared input
sets `ndiag=1` and history precision 17 to preserve dt/cycle evidence.

The historical evolution used **four nodes / 48 ranks**. New 24- and 96-rank
native runs test different decompositions. Local CPU tests with two/eight ranks
do not establish correctness at 24/96 ranks; the latter also creates many ranks
without surface faces. Do not label a successful native run as the historical
binary or decomposition.

## Saved-state GPU replay before fresh evolution

As of **2026-11-08 06:33:35 UTC**, the surviving reference binary remains:

`/lus/flare/projects/CompactBinaryMerger/hzhu/scri-resolution-20261107/fastflow_diagnostics`

SHA-256 `ffc8cd31909776872f0490466b26fbaeda378e072478e68e24ec9c7aaa2f0701`.
Its required `libcurrent_spherical.so` still has SHA-256
`ad4c6321b18a44ee89e42cdd4307e760fe59506ec0293a655e97416811acff2d`, and `ldd`
resolves every dependency under the pinned module. The step-3551 checkpoint and
its exact expected-norm row in `campaign-constant20m/horizon-analysis/snapshots.txt`
are present. The subsequent authorized two-node replay completed successfully, with maximum
full/fixed-outer norm difference `8.6736173798840355e-19`, all eight strict checks
passing, and a complete horizon after 149 iterations. Its immutable copies passed
before/after checksum checks. See
[evidence/aurora-reference-replay-two.json](evidence/aurora-reference-replay-two.json)
and `build-native/aurora-reference-replay-two` for the complete collected output.
This does not yet establish native GPU correctness. The historical wrapper-source and
launch-time binary provenance gaps in `N12-reference.md` still apply.

The completed isolated one-snapshot replay copied the saved checkpoint, binary,
custom shared library, input and configuration into private `reference/`,
verified their pinned hashes, rewrote only the snapshot row's checkpoint path,
and used a new output directory. The expected radius, time, and all eight
full/outer RMS values were retained verbatim. The old executable's interface is:

```sh
mpiexec -n "$RANKS" -ppn 12 --cpu-bind=list:"$binding" gpu_tile_compact.sh \
  "$TASK_ROOT/reference/fastflow_diagnostics" \
  "$TASK_ROOT/reference/central_smr20.in" \
  "$TASK_ROOT/reference/snapshot-3551.txt" "$CASE/reference-replay"
```

The replay uses private input/config copies. The old binary has an absolute
shared-library rpath; the executed script prepends the private `reference/`
directory to `LD_LIBRARY_PATH` so the hash-matched custom shared-library copy is
used. `ldd` and before/after checksums are preserved. Do not run the production PBS script: its output path is inside
the existing campaign. A replay loads saved N12 state and computes diagnostics;
it does not resume evolution or touch N24/N36.

The exact fixed-outer cut in the surviving old N12 source is
`sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) > .94` on the **device**, with nominal-cell
volume weights and zero overlap weight. Sources:

- `scri-ko01-20261106/source/src/z4c/hyperboloidal/hyperbolic_composite_device.hpp`,
  lines 449, 454, 499 (original exterior norm).
- `scri-resolution-20261107/source/src/z4c/hyperboloidal/hyperbolic_composite_device.hpp`,
  lines 450, 458, 503 (fixed-cut replay norm).
- `scri-resolution-20261107/source/tst/hyperbolic_shift_3d/endpoint_policy/fastflow_diagnostics.cpp`,
  lines 32–38: `.94` fixed cut and strict replay check.

Current CPU native and independent legacy constraints agree pointwise. The
historical strict CPU replay failure is preserved: full H RMS differs by
`5.81e-11`, and fixed-outer norms differ more because the strict `.94` cutoff
is sensitive to coordinate rounding. Complete CPU/Aurora point dumps now show
118 positive-weight cutoff flips; every flip has changed coordinates, while
host/device radius bytes agree within each platform. Swapping only the mask
reproduces nearly all outer-norm differences. This does not assign the separate
full-H arithmetic discrepancy or explain the eight-step trajectory difference.
The unchanged strict native replay passes on Aurora (24 ranks), including all
full/fixed-outer norms, history, horizon and read-only checks. See
[evidence/native-saved-state-cross-architecture-attribution.json](evidence/native-saved-state-cross-architecture-attribution.json)
and [evidence/aurora-native-saved-replay-two.json](evidence/aurora-native-saved-replay-two.json).

## Missing gates and staged acceptance

1. Freeze the diagnostics milestone, finish local constraint/history/restart
   gates, record main/Kokkos/source hashes, stage privately, and complete a fresh
   native SYCL build. Check that real execution is on the expected GPU tiles and
   that MPI device-buffer exchange works. The native Aurora build, saved-state replay and eight-step smoke have passed
   on 24 ranks; complete state comparison and 96-rank validation remain.
2. Resolve the saved-state CPU/GPU diagnostic discrepancy through the unchanged
   original binary replay and native saved-state constraint/sampler oracles on
   the same Aurora toolchain. Retain `5e-11*(1+abs(expected))` for the existing
   full/fixed-outer replay gate. Inspect threshold membership explicitly.
3. Adapt **test-only** oracle launchers for Aurora. Current
   `constraint_oracle.py` and `horizon_sampling_oracle.py` hard-code
   `/usr/bin/c++`, omit native SYCL compilation flags, and admit only two/eight
   ranks. They cannot be run unchanged as 24/96-rank GPU tests. Recover compiler
   and options from the native CMake build, preserve source/dependency hashes,
   support tile binding, and keep their independent legacy reference arithmetic.
4. Compare initial data, one complete RHS, one full step, and eight full steps
   at 24 and 96 ranks, using all owned bulk/scri fields and chi ledgers, poisoned
   unused data/corners, zero-face owners, pointwise maxima, volume-weighted RMS,
   worst coordinates, and per-stage compatibility/clock records. Require the
   adopted `1e-12+5e-10*abs(reference)` short gate, separate
   `1e-14+5e-10*abs(chi_reference)` chi gate, and compatibility <=1e-7. Keep
   bitwise comparisons wherever arithmetic order is preserved.
5. Only then run fresh 3551-step N12 at both node counts. Decode native restarts
   and match complete arrays by leaf/physical coordinate to the pinned old
   `fields.bin`/`boundary.bin`; do not compare incompatible restart formats
   directly. The long-run comparison adapter and GPU active-floor evidence
   remain to be completed. Apply the unchanged adopted long-state, separate chi,
   common-time RMS history, horizon-completeness/mass/radius, floor-candidate,
   correction, and ledger gates. Report deviations without projecting either
   accepted state to hide them.
6. Only after correctness passes, prepare separate performance windows with
   identical workloads, detailed profiling disabled, warmup and repeated
   bounded complete-RK4-step windows, maximum-rank fenced elapsed times, and
   startup/history/checkpoint costs separated. The prepared N12 correctness
   scripts do not claim performance or the goal's N24 two-to-eight-node target
   `T2/T8 >= 3` (75% efficiency). N24 private-checkpoint performance is a later
   milestone; this plan never restarts its production campaign.

## Current hold and unchanged GPU control

The actual eight-step native GPU checkpoint passes the initial-state comparison
but **fails** the unchanged short gate against the independently evolved local
CPU reference (381192 bulk and 13994 scri values; maximum absolute differences
about `4.89e-9` and `1.32e-9`). Preserve that failure. It is not yet established
whether this is a migration difference or a cross-architecture evolution effect.
The failure initially held further native runs. The subsequent unchanged
GPU-to-GPU comparison passed the original short/chi gates: initial fields are
bitwise equal, and step-eight maxima are `5.551115123125783e-17` in the bulk and
`4.336808689942018e-17` on scri, with available aggregate chi diagnostics equal.
All 688176 bulk coordinate components and ten resolved gauge/profile identity
doubles agree exactly. The failed GPU-to-CPU comparison remains recorded; no
tolerance changed. Full evidence is
`build-native/aurora-native-vs-original-gpu-eight-v3/results.json`.
The full-3551 evolution remains gated on the remaining correctness tests.
The unchanged historical `evolve-static` executable was copied privately and
verified against its original SHA-256
`494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753`.
Its original entry-point/control/header hashes match the recovered historical
build manifest. Job **9051418** completed the authorized two-node/24-rank fresh eight-step
control (PBS finished, exit 0); the only input change was `nlim=-1` to `nlim=8`.
All 61 downloaded files passed individual hash/size checks. Evidence is in
[evidence/aurora-original-eight-control.json](evidence/aurora-original-eight-control.json).
Its complete fields,
boundary, compensation-bearing restart, coordinates, per-step diagnostics, and
aggregate chi policy are being compared directly to native GPU results. The old
binary does not export per-cell ChiRecords, and no such records will be claimed.

Local `run-native-validation-eight.sh` and `native-validation-eight.pbs` are
prepared and shell-syntax checked for the same frozen native executable and
adapter on eight nodes/96 ranks. They retain the same strict replay-first gate,
complete 96-file canonical dump check, input, timestep, eight-step smoke, and
immutable provenance checks. After the matched 24-rank PASS, the three launcher files were staged separately
from the immutable source and submitted as **9051435**, eight nodes/96 ranks,
30 minutes. The job additionally runs the unchanged original GPU96 eight-step
control after native replay and smoke pass. No full-3551 run or profiling is
included. All binaries and physics are identical to the two-node tests;
matched native96/original96 and native24/native96 comparisons follow the run.

## Conditional full-run continuation schedule

This is planning only while the 96-rank and stage/poison gates remain pending. A fresh full run must
start without a restart argument, not from the off-cadence eight-step checkpoint.
The latter includes a final-output horizon search and an output-clock increment
at step eight that do not belong in the original common-time history.

Ending a common-time segment using only `nlim` with `tlim=2` would write both its
regular scheduled output and `Driver::Finalize` output at the same cycle,
advancing normal output clocks twice. Instead use the existing native fixed-dt
policy with each segment's **nominal common diagnostic time** as `time/tlim`,
and its required cumulative cycle as `time/nlim`. In `src/mesh/mesh.cpp:695`,
hyperboloidal evolution explicitly bypasses ordinary timestep clipping;
`src/driver/driver.cpp:653` suppresses in-loop time output at `time>=tlim`, and
`Finalize` writes the single endpoint record. `Initialize(res_flag=true)` at
line 524 emits no extra initial output. `FastFlow::Find` saves `last_a0` and its
found flag into the parameter header before the normal restart writer runs.
Neither output counters nor the horizon seed need resetting or overriding.

A float64/float32 simulation of these exact existing stopping/output conditions
matches all eleven preserved common snapshots, including cycle zero. Results and
wall-time estimates are in
[evidence/native-n12-continuation-plan.json](evidence/native-n12-continuation-plan.json).
The nominal limits below are printed with sufficient digits to preserve the
normal accumulated `1/6` output clock; the evolved time still advances by the
unchanged complete nominal dt.

| Segment | Start → end cycle | `time/tlim` | Actual final time/M | Next restart number |
| --- | --- | --- | --- | --- |
| 1 | 0 → 356 | 0.16666666666666666 | 0.5013640958963977 | 2 |
| 2 | 356 → 711 | 0.3333333333333333 | 1.0013198656807232 | 3 |
| 3 | 711 → 1066 | 0.5 | 1.501275635465034 | 4 |
| 4 | 1066 → 1421 | 0.6666666666666666 | 2.0012314052494036 | 5 |
| 5 | 1421 → 1776 | 0.8333333333333333 | 2.5011871750337735 | 6 |
| 6 | 1776 → 2131 | 0.9999999999999999 | 3.0011429448181435 | 7 |
| 7 | 2131 → 2486 | 1.1666666666666665 | 3.501098714602395 | 8 |
| 8 | 2486 → 2841 | 1.3333333333333333 | 4.001054484386646 | 9 |
| 9 | 2841 → 3196 | 1.5 | 4.5010102541708985 | 10 |
| 10 | 3196 → 3551 | 1.6666666666666667 | 5.0009660239551499 | 11 |

For segment k>1, use only the preceding **validated common-time** native restart
with the ordinary `-r` interface, set the next cumulative `nlim`/nominal `tlim`,
and preserve every embedded physics, fixed-origin horizon and output parameter.
Retain all block costs equal to one. Use a unique private output directory per
segment and combine history records by their actual time/cycle. Confirm the
normal restart counter, both output clocks, the horizon seed and complete payload
at every handoff. No acceptance may rely on a wall-time-terminated partial result.

Omit the software `-t` stop for this specific segmented campaign: otherwise a
wall stop would call `Finalize` and create an off-cadence horizon search. Bound
the allocation externally (one-hour queue limit, e.g. 3420-second hard timeout).
If a segment cannot reach its prescribed cycle, retain its failure evidence and
return to the previous validated common-time checkpoint; do not admit any partial
checkpoint or add an off-cadence continuation point.

The seven non-final smoke intervals range from 4.704 to 5.767 seconds. Simple
extrapolation gives approximately 27.8–34.2 minutes of evolution for a 355/356-step
segment. A planning allowance of 1.25 times the observed maximum plus 180 seconds
for startup/diagnostics/I/O gives about **45.7–45.8 minutes per segment**. This is
an uncertain queue-planning estimate, not a benchmark or guaranteed upper bound.
Use one common interval per job initially; reassess only from completed evidence.
The schedule preserves the exact final cycle 3551 and dt, independently of
whether the available wall time proves sufficient. No full segment is authorized
to bypass the unresolved short-state or eight-node correctness gates.


## Eight-node bundle completion and held continuation runner

PBS **9051435** finished with exit zero on eight nodes/96 ranks in 3m03s.
The strict saved3551 replay passed; the frozen native executable and unchanged
original GPU executable each completed eight steps with the pinned dt. All368
artifact files were transferred and verified against remote hashes and sizes.
Independent complete native96/original96 comparison passed the unchanged short
and chi gates. Native24/native96 all31 fields and all8 ChiRecords are byte-identical
at initial and step8, with exact actual bulk coordinates. First-RHS/accepted1
matched-GPU and stage/poison gates remain pending. The recorded
short timings include startup, diagnostics and I/O and are not performance gates.
See [bundle evidence](evidence/aurora-native-eight-node-bundle.json).

The locally prepared `aurora/run-common-segment.sh`, `common-first-two.pbs`,
and `common_segment_audit.py` implement the first two-node common interval
0→356 and the described restart handoffs. No common segment has been submitted.
The script requires the short96 and stage/poison evidence, checks frozen runtime
inputs before and after execution, preserves all block costs as one, and uses
only `-r` for later segments. The audit checks complete checkpoint structure,
exact time/dt/topology/physics identity, both normal output clocks, restart
counters, retained horizon seed, and the expected number of history/horizon
records. It explicitly leaves full fields, chi and reference diagnostics for
the independent comparator before continuation admission. Header/cadence positive
and negative unit controls passed; an actual common interval is still unexecuted.

The current native N12 migration goal supersedes the initial N24 benchmark scope.
The existing N24 production campaign remains read-only; this plan does not add
an independent N24 benchmark milestone.


## First-stage and poison validation

Three isolated test executables built successfully with the verified Aurora
compiler: the native Driver stage observer; a read-only accessor of the exact
pinned predecessor Evolution class; and a metadata-only native/legacy coordinate
mapper. Their builders reuse the actual native or archived original compile/link
commands and preserve runtime source/object/executable bytes. Build provenance
is recorded in [the observer build evidence](evidence/aurora-stage-trace-builds.json).

PBS **9051466** was submitted to debug on two nodes/24 ranks with a30-minute
allocation. It runs the one-rank coordinate mapper, native clean and poison
eight-step observers, the predecessor eight-step accessor, and the unchanged
original binary for a fresh one-step control. The accessor exports the actual
first RK-stage rates retained in `rates[0]`/`edge_rates[0]`, without an extra
evaluation inside RK. It also records accepted states separately from its
final diagnostic-snapshot state. The latter is checked against the existing
unchanged eight-step control; the unchanged one-step control supplies a separate
check of that diagnostic phase. All scientific comparisons remain pending until
artifacts and their complete coverage are verified.

The analogous eight-node/96-rank job is authorized only after the24-rank job
becomes terminal, keeping the total active allocation at or below eight nodes.
The full3551-step trajectory remains held until both matched short protocols and
clean/poison/cross-rank stage checks pass.


The first observer bundle,9051466, stopped with exit139 during metadata-helper
teardown before any evolution ran. Its complete mapper outputs and52 provenance
files are retained as failed, unadmitted evidence. The helper omitted coordinates
allocation but left the corresponding owned pointer uninitialized; explicit
null initialization fixes only that test lifetime. Mapping formulas and all
runtime objects remain unchanged. The corrected metadata helper built as
`a14333e4…`; fresh retry **9051476** uses new output directories and the same
native/legacy observer executables. The first failure is not counted as a passed
geometry or evolution gate.

The held common-segment runner now calls `aurora/longrun_admission.py`, which
parses the actual saved-replay, full-state, matched first-stage, clean/poison and
cross-rank report schemas. It requires the actual SYCL24/96 decompositions,
complete bulk/scri coverage,31 fields/all8 histories,26 observer label/region
sets,32 stage records, and frozen executable/command bindings. It also pins the
admitted initial checkpoint's exact path and SHA. Available96 full-state and
cross-rank reports pass the parser; wrong-rank and missing-record negative
controls are rejected. Complete admission remains pending the actual trace gates.


Retry **9051476** completed normally in5m43s. The corrected metadata helper
passed after teardown. Native clean and poisoned eight-step traces passed their
complete bytewise comparison: all26 label/region combinations and32 stage
records, including the first RHS and all31 fields/eight chi records. The exact
pinned predecessor accessor and immutable one-step binary control also completed.
All1675 output/provenance files were transferred and hash/size verified before
scientific comparison. [Two-node trace evidence](evidence/aurora-stage-trace-two.json)
retains the distinction between operational/poison success and the pending
matched first-RHS/accepted-state scientific admission.

After9051476 became terminal, **9051485** was submitted on eight nodes/96 ranks
using the same native/legacy observer binaries and corrected mapper. No two-
and eight-node allocations overlap. Full3551 remains held. The present trace
exports stage clock/compatibility and maximum RHS, but not CFL; a separately
validated observer of the original main is planned to close that explicit
protocol requirement before long correctness runs. Its timing will be excluded
from performance measurements.


The corrected trace bundles **9051476** (24 ranks) and **9051485** (96 ranks)
now pass the complete matched scientific comparisons. First-stage RHS, accepted
steps 1 and 8, all chi records, and actual coordinates satisfy the declared
gates; all 26 native dump label/region combinations and 32 stage records are
bitwise identical between 24 and 96 ranks. The earlier failed metadata-helper
job remains retained separately.

The observed original-main executable built as `f2c2e6cd…`, using the unchanged
numerical objects and exactly reversible include/Install/Finish additions to
main. Its original compiler flags are preserved. Two-node PBS **9051515** runs
a fresh eight-step case and a fresh two-step prefix; the subsequent serialized
eight-node case will run fresh eight steps and restart the 24-rank prefix to
step 8. Exact stage clocks, compatibility, and CFL are recorded without
re-evaluating the RHS. The full common-time run remains held until its actual
GPU state, diagnostic, clock/CFL and rank-changing restart comparisons pass.
These observer timings will not be used as performance measurements.

All eleven historical N12 common snapshots (initial through step 3551), their
SP1 `restart.bin` files, complete fields/scri data, chi-policy records, and the
full historical `steps.csv` were copied into the private reference directory.
Every source, private-copy and downloaded hash matches. The original campaign
was read only. The independent binder verifies each field snapshot against its
SP1 payload and the common topology/physics identity.


Observed-main jobs **9051515** (24 ranks) and **9051527** (96 ranks) both
completed normally. The complete independent GPU qualification passes: all
31 fields and eight history values in every padded checkpoint record are
bitwise equal to the normal executable at initial and step 8; ordinary
history, horizon summary and shape outputs are byte-identical. The 24- and
96-rank states and all 32 stage observations are exact across ranks. Restarting
the verified 24-rank step-2 checkpoint on 96 ranks reproduces the continuous
step-8 state and joined stage observations exactly. Every clock/CFL maximum
for the eight steps is bitwise equal to the corresponding unchanged original
GPU control. Restart diagnostics satisfy the declared gates, with the extra
prefix-final horizon search explicitly retained in that comparison. See
[observer qualification evidence](evidence/native-step-observer-aurora.json).

The first fresh common-time segment may now be submitted only after the full
31-report short-admission descriptor and frozen runtime checks pass on Aurora
and a fresh queue check confirms resource limits. It stops at cycle 356; no
later continuation is automatically admitted by the short tests.


The full 31-report descriptor passed on Aurora, followed by unchanged runtime
source/object/executable checks and a fresh queue check. The first fresh common
segment was submitted as **9051541** on two nodes/24 ranks, with a one-hour
allocation and an external 3420-second process bound. It advances only 0→356;
no software wall-clock termination or off-cadence checkpoint is used. The
endpoint must pass complete field, chi, history, horizon, stage-clock and CFL
comparisons against the verified historical step-356 reference before any
further continuation is admitted. This submission does not establish a long
trajectory result or a performance measurement.


The complete first 356-step comparison and independent review both pass.
All owned bulk/scri fields satisfy the declared long-run gates, all 1,424
stage observations pass clock/CFL and compatibility checks, and both ordinary
diagnostic times pass. The initial null-aggregate reporting failure is retained;
its fix changes reporting only. Native chi records are decoded and finite;
historical aggregate differences are assessed under the declared threshold
policy, without claiming unavailable historical per-cell record equivalence.

Segment 2, **356→711**, was submitted exactly once as **9051587**, using two
nodes/24 ranks and the original restart entry. Its preflight binds the exact
accepted checkpoint, complete science report and independent review hashes,
then verifies normal output clocks/horizon seed and all frozen runtime inputs.
The timestep, output cadence and numerical executable remain unchanged. A
fresh queue check showed only the three existing held campaigns. Segment 3
remains held until the complete cycle-711 comparison and review pass. See
[second-segment submission evidence](evidence/native-n12-second-common-segment.json).


Segment 2 completed normally at cycle 711 as PBS **9051587** (terminal F/0).
All 90 archived files, including the complete 1,420 new stage observations,
checkpoint, diagnostic outputs and exact predecessor admission, verified by
size and SHA256 after download. The checkpoint SHA is
`2bc744ea139041c33bf2275e756515050ebbc5d33581a7990bc8fcc9bf6d5b89`.
The complete cycle-711 scientific comparison and independent review both
pass. All endpoint fields, cumulative 2,844 stage records, and histories at
0/356/711 satisfy the unchanged gates. Segment 3 remains held while the
separate AMR qualification runs. The recorded allocation wall time is
correctness-run planning information, not a performance measurement.


The isolated coherent AMR candidate and both test helpers built successfully
with the verified Aurora compiler and frozen numerical flags. Their complete
source/object checks pass, and the frozen N12 build is unchanged. After the
cycle-711 comparison and independent review passed, the reviewed two-node
AMR qualification was submitted once as **9051631** (24 ranks). It runs the
field-transfer lifecycle, four-cycle adaptive evolution, and ordinary
two-cycle restart prefix at the separately justified AMR timestep `.0002`.
This is a distinct adaptive admission test, not a change to the N12 trajectory.
The eight-node allocation is held until the two-node allocation terminates
successfully and the actual prefix checkpoint is verified. Complete GPU
AMR science remains pending; no performance claim is made.


The two-node AMR bundle **9051631** completed F/0. All three launches passed
operational coverage and the post-run source/object guards; its ordinary
cycle-2 checkpoint was parsed and its duplicate final payload verified. All
453 collected files match their remote hashes and sizes. After a fresh queue
and provenance check, the serialized eight-node bundle was submitted once
as **9051645** (96 ranks). It repeats both helpers, runs a continuous four-cycle
ordinary evolution, then restarts the admitted 24-rank cycle-2 checkpoint to
cycle 4. Complete cross-rank AMR field/history and restart science remains
pending until those outputs are collected; no later N12 segment is submitted.


Both AMR allocations completed F/0 with all seven launch-level checks and
frozen-input guards passing. The complete 24/96 AMR comparison is performed
offline; it does not gate the already admitted fixed-mesh N12 checkpoint.
After the eight-node allocation ended and a fresh queue/provenance check
passed, the third N12 segment **711→1066** was submitted once as **9051654**
on two nodes/24 ranks. Its preflight binds the complete cycle-711 science and
independent review, the exact checkpoint and output/horizon clocks, and the
unchanged normal/observed N12 binaries. It uses the original restart entry
with no timestep override. No segment beyond 1066 is admitted yet.


The complete isolated GPU AMR qualification now passes for all three cases:
field transfer, four-cycle adaptive evolution, and ordinary rank-changing
restart. All 31 fields and eight history-record values in the canonical
snapshots are bitwise equal between 24 and 96 ranks. The 24-rank prefix
restarted on 96 ranks reproduces the continuous 96-rank payload exactly,
including all 26,336,232 padded scalars per checkpoint comparison. The
independent review passes. This qualifies the separate coherent candidate at
`.0002` for the bounded transfer/evolution/restart scope; it does not establish
long adaptive stability or performance. See
[complete GPU AMR evidence](evidence/native-amr-coherent-gpu-validation.json).

The unchanged fixed-mesh N12 segment **9051654** is running; the observed
progress at cycle 720 has 36 new stage records and the exact
original timestep. Cycle 1066 and any later continuation remain unadmitted
until complete endpoint science and independent review.


Segment 3 completed as **9051654** (F/0), and all 91 collected files verified
against their remote hashes and sizes. Complete cycle-1066 science and the
independent review pass, including all endpoint fields, cumulative 4,264
stage observations, and the completed diagnostic prefix. The admitted
checkpoint SHA is
`745a158d2eeb4c05111d1a26a5e971c955439ca3360ce12e9078aaf0fe336143`.

The reviewed reusable preflight passed on Aurora for the fourth segment,
binding that checkpoint, the complete science report, independent review,
ordinary output/horizon clocks, and unchanged runtime. After confirming the
previous job was terminal and only the three existing held campaigns remained,
**1066→1421** was submitted once as **9051691** on two nodes/24 ranks. This uses
the original restart entry and timestep; no segment beyond 1421 is admitted.
See [fourth-segment submission evidence](evidence/native-n12-fourth-common-segment.json).

Segment 4 completed as **9051691** (F/0), and all 91 collected files verified
against their remote hashes and sizes. Complete cycle-1421 science and the
independent review pass: all 7,137,936 endpoint fields, 1,842,048 native chi-record
values, 5,684 cumulative stage observations, and all five common diagnostic
times are checked. The 37,968 floor events in eight cells match the historical
counts; the correction-sum difference `2.168404344971009e-19` is retained. The
admitted checkpoint SHA is
`3ceb2d9f1f730cca2d1e4b85cf46a083ce619d6c5436b47f5ebedb1e27560115`.
See [cycle-1421 science](evidence/native-n12-common-04-science.json) and
[independent review](evidence/native-n12-common-04-independent-review.json).

After fresh queue and runtime checks, **1421→1776** was submitted once as
**9051736** on two nodes/24 ranks. Its preflight binds the exact admitted
checkpoint, complete science report and independent review, output/horizon
clocks, and unchanged runtime. It uses the original restart entry and timestep.
No segment beyond 1776 is admitted; six later common endpoints and the full
3551-step result remain unverified. See
[fifth-segment submission evidence](evidence/native-n12-fifth-common-segment.json).

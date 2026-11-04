# Reproduction and provenance

The baseline is commit `c16a7c212acb00e9f963cb9a9f42d5d886777a7e`,
with Kokkos `6739bc623081648af9e752b616d9671527922cbf` (4.7.2).
All source archives, overlays, frozen executables and states remain in the
isolated local `build-performance/` directory and in
`hzhu@aurora.alcf.anl.gov:/home/hzhu/hyperboloidal-performance-scaling-20261103/`.
The local worktree is
`/Users/hz0693/.codex/worktrees/hyperboloidal-performance-scaling/hyperboloidal`.
Neither original checkout is a build/output destination.

The archived manifests identify bytes, not just nominal commits. Early source
snapshots were taken between instrumentation commits. The local frozen `m0-v2.bin` is the committed M0 source `b4cfdfd5b`.
To replay the Aurora variants exactly, use
`control-source-v2.tar.gz` plus `m1-overlay.tar.gz` for M0/M1a; add
`m1b-overlay.tar.gz` for the rejected radial experiment, or `m4-overlay.tar.gz`
for the direct-setup experiment. Do not combine the radial and setup overlays.
Each overlay has a per-file SHA256 manifest. The baseline source manifest covers
AthenaK and Kokkos. The branch history preserves the rejected mask at
`fda41dba1`; `81226ce6d` removes it from the final source.

## Build and run

Use a fresh build/output directory for each replay. CPU configuration was:

```sh
cmake -S SOURCE -B BUILD -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
  -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON \
  -DKokkos_ENABLE_SERIAL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build BUILD -j4 --target hyperbolic_shift_3d_amr_setup \
  hyperbolic_shift_3d_amr_setup_contract
mpiexec -n 2 BUILD/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_setup \
  12 wave 6 .00025 FRESH_PREFIX - dynamic
```

`hyperbolic_shift_3d_amr_evolution` remains the instrumented control;
`hyperbolic_shift_3d_amr_setup` enables direct final halo construction.
The diagnostic-transfer target is an opt-in experiment, not a recommended
optimization. Its source remains available for targeted follow-up. The radial
mask is absent from final source because it did not pass the performance gate.

The exact Aurora build flags are in `scripts/build-control.sh` and
`scripts/build-m1.sh`. Isolated replay builds use
`tst/hyperbolic_shift_3d/performance/build_variant.py`: it reads the frozen
compile database/link command, substitutes only the standalone source/output and
experiment definition, and records the resulting argument vectors. The complete
native archive is reused because none of its compiler dependencies includes the
changed host-model headers. Local main-Z4c validation similarly compiles the
original `main.cpp` and links the same complete native archive; the retained
command record makes this explicit.

Every performance plan freezes binaries and records commands, hashes, return
codes, elapsed times and checkpoint hashes. `scripts/job1.pbs` and `job2.pbs`
are the executed bounded scripts; their JSON plans and per-run receipts are the
exact launch specification. Source/plan/binary hashes were recorded before
submission. Jobs run sequentially, with at most two nodes and a reserve before
walltime. The runner itself never submits a job.

Only `/usr/bin/python3.11` supports the submit helper on Aurora. One attempted
job-2 helper invocation used the older default Python, failed before reaching
qsub, and left its `preparing` receipt. Scheduler reconciliation showed no task
job before the explicit Python-3.11 invocation submitted the single job. Both
receipts are retained. The earlier Apple-metadata source-tar extraction failure
is also preserved; `control-source-v2.tar.gz` is the verified clean archive.

`scripts/qualify-local.py` and `scripts/run-m4-local.py` preserve the local
orchestration. Their absolute workspace paths are evidence; substitute a fresh
workspace when replaying. Benchmark runner copies under each output directory
are immutable, including the early CPU runner preceding the later harness
extensions. See `commands.json`, compile databases, replay `.commands.json`
files and build logs for the actual commands rather than inferring a build from
the latest source tree.

## Analyze and retrieve

```sh
python3 tst/hyperbolic_shift_3d/performance/analyze.py RUN_DIRECTORY --output SUMMARY.json
python3 tst/hyperbolic_shift_3d/performance/check_histories.py RUN_DIRECTORY --output HISTORIES.json
python3 tst/hyperbolic_shift_3d/performance/plot_results.py EVIDENCE_DIRECTORY PLOT_DIRECTORY
```

The plot command requires NumPy/Matplotlib; the validation environment also uses
pytest. Dependency-install output is retained. The normal timing runs load no
profiling library. Optional fenced profiling uses the recorded separate cases,
not the timing pairs.

`scripts/collect_remote.py JOB_NUMBER` packages small receipts, logs and JSON
results and creates a SHA256/size manifest for large checkpoints/captures.
Retrieve these from the isolated remote root using an existing authenticated
SSH connection, then verify the manifest before reading a state. It never deletes
or modifies a state. Local large-artifact manifests cover the corresponding
local frozen binaries, source archives and checkpoints. Source files in a replay
archive must match the per-file overlay/control manifests. No artifact is
relabelled as coming from the final build.

Final source adds sparse-peer and sampled-normal work counts (performance schema
2), and fixes optional profiler free attribution; these instrumentation changes
postdate the frozen M4 timing binary. Scientific source, setup construction and
its compile definition are unchanged. The final CMake-built target receives a
separate bitwise smoke check. Early M0 generation `retired.leaves` labels reflect
the already-regridded native mesh; use its unchanged recorded work/model sizes,
or the corrected M4 generation labels, to identify the retired generation.

The final fixed-layout CPU M4 pair is explicitly replaced because the analysis
dependency install overlapped its candidate at 00:16:05–00:16:08 UTC. Both
original runs remain in `cpu-m4-bench`; the replacement is in
`cpu-m4-replacement`, and `cpu-m4-clean/selection.json` records the derived
analysis selection. `scripts/replace-cpu-pair.py` runs only after all local
qualification and final checks finish, using the identical frozen executables.
No numerical mismatch or unfavorable result was the reason for exclusion.

The first supplemental local build encountered a stale CMake generated target
list (`No rule to make target hyperbolic_shift_3d_amr_setup`). No numerical test
failed at that point. `scripts/resume-qualification.py` explicitly regenerated
the same configured build and resumed at compilation; completed AMR suites were
not repeated. The original failure and subsequent command records are retained.

For job 2, `scripts/collect_remote_v2.py` additionally retains remote compile
databases, CMake caches, replay command vectors and source/native-library hashes.
`evidence/package-sha256.json` hashes the delivered evidence, plots and report.
The artifact verifier is an audit of this retained run selection, including its
documented harness-failure receipt; a future replay should retain its own receipts
and must not relabel them as these measurements.

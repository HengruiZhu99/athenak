# Reproduce the direct lapse-gradient evidence

The campaign is finished with Gate 2 FAIL. The commands below document reproduction;
they do not authorize automatic downstream launches or resuming the failed state.
Run analysis from the repository root. `ARTIFACTS.sha256` paths are repository-relative.

## Recreate figures and summaries without another simulation

Python requirements are recorded in `analysis-requirements.txt` (Python 3.9 plotting
environment). The reducer and its tests use only the Python standard library.
The local plotting interpreter used was `.venv-bbh-plots/bin/python`.

```sh
E=qualification-runs-20260905/direct-lapse-gradient
A=analysis/pc_gh_regular_extension
python3 "$A/test_direct_lapse_monitor.py"
.venv-bbh-plots/bin/python "$A/analyze_direct_lapse_stress.py" \
  "$E/evidence/baseline" "$E/evidence/corrected" --output "$E/analysis"
.venv-bbh-plots/bin/python "$A/plot_direct_lapse_operations.py" \
  "$E/evidence/baseline" "$E/evidence/corrected" --output "$E/analysis"
.venv-bbh-plots/bin/python "$A/plot_direct_lapse_correctness.py" "$E"
```

The committed compressed completed-step monitors and stage/operation envelopes
are sufficient for these commands. Histories and segment logs supply the native
norms and terminal decisions. `comparison.json` has common-time RMS, scalar
extrema, locations, minima, and per-region growth flags. Figure bytes may depend
on fonts/library versions; numerical tables are the reproducibility target.
The completed monitor has raw `t_step`; for the frozen RK3 input, stage=3,
operation=-2, phase=before is labeled with physical time `t_step+dt`.
Half-M stage/operation envelopes intentionally use step-start time bins.

## Original Della evidence

SSH host: `hz0693@della-vis1.princeton.edu`.
For this local session the existing control connection is selected by
`-o BatchMode=yes -o 'ControlPath=/Users/hz0693/.ssh/codex-control/%C'`.

Corrected campaign root:
`/scratch/gpfs/FPRETORI/hz0693/pcgh-direct-lapse-20260905`.
The executable is `source/build-direct-cuda/src/athena`; original run outputs
are in `stress/core256-R16`, including raw hybrid/reduction monitors, field
outputs, checkpoints and the saved research sources. The terminal binary/input
hashes are in `terminal-audit.json`. `raw-artifacts.json` inventories and hashes
all original baseline/corrected run files, including checkpoints and field outputs.

Baseline run:
`/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/core-hybrids/R16/core256-R16`.
Baseline source/build provenance is in its `provenance.json`, `source-diff.patch`
and `CMakeCache.txt`. Its original source was the monitor-source checkout under
that campaign root. Do not identify corrected source by its base commit alone:
the patch and full 319-file source manifest are part of the identity.

Both compact evidence directories are also preserved under the corrected root's
`evidence`. To regenerate reductions from immutable original monitors on Della:

```sh
R=/scratch/gpfs/FPRETORI/hz0693/pcgh-direct-lapse-20260905
B=/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/core-hybrids/R16/core256-R16
python3 "$R/source/analysis/pc_gh_regular_extension/reduce_direct_lapse_monitor.py" \
  "$B/core256-R16.pcgh-reduction.csv.hybrid.csv" "$R/reproduced-evidence/baseline"
python3 "$R/source/analysis/pc_gh_regular_extension/reduce_direct_lapse_monitor.py" \
  "$R/stress/core256-R16/core256-R16.pcgh-reduction.csv.hybrid.csv" \
  "$R/reproduced-evidence/corrected"
```

The gzip header timestamp can differ; compare decompressed CSV content and the
source SHA256 recorded by the monitor audit. Baseline/corrected raw monitor hashes:

- `69fd76396a1550b65f3f78f23c6f44e0e3325cc259a723e8b7e6a29c2b428486`
- `ce675988a9bd70be7f524529ff8fc506e3bb85cb3de8dfeb607c1ae07fff4fc6`

The reducer checks file size/mtime before and after its two passes and refuses
changed inputs. It audits rollback epochs before pairing operations. Do not use
it on a monitor still being written.

## Build and correctness tests

Production source is implementation commit `5a9230e3` in this branch, with the
same problem generator/Kokkos sources as the saved baseline. `implementation.patch`
and `local-source-sha256.json` identify the tested source; terminal audit rechecked
all 319 production files. The later analysis edits do not change production source.

On Della, source `/home/hz0693/athenak_env` before enabling shell nounset.
`build-provenance/CMakeCache.txt`, `build-option-comparison.json` and `toolchain.txt`
preserve the exact build options/environment. The existing frozen build can be
rechecked with `cmake --build source/build-direct-cuda -j 4` from the remote root.
Core configure options for rebuilding in a new source checkout are:

```sh
cmake -S . -B build-direct-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_AMPERE80=ON -DAthena_ENABLE_MPI=ON \
  -DPROBLEM=../../analysis/pc_gh_regular_extension/production_oracle
cmake --build build-direct-cuda -j 4
cmake -S analysis/pc_gh_regular_extension/direct_lapse_test \
  -B build-direct-lapse-test \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DKokkos_DIR="$PWD/build-direct-cuda/kokkos"
cmake --build build-direct-lapse-test -j 4
./build-direct-lapse-test/direct_lapse_test
```

For the CPU matrix, use the same test CMake project with a host compiler and
`Kokkos_DIR` pointing to the local `build-regular-extension/kokkos` directory.
No GPU is needed for that test. The executable prints the complete 12-case CSV.
The saved CSVs, build/configure logs and `gate1.json` are committed.
GPU execution must follow the established driver/host policy and current resource
availability; compiling alone does not allocate a GPU.

`run-correctness.py` is the actual remote-root driver for the 39 frozen zero-step
projection cases in `inputs/oracle`. It calls `cuda_driver.py` and the independent
`verify_hybrid_projection.py` with `direct_lapse=True`; existing oracle outputs
and per-case verification records remain at the remote root in `oracle`.
`projection-results.json` and `projection-checks.log` preserve all outcomes.
`verify_direct_lapse_diagnostics.py` independently checks the diagnostic max/L1
values against those saved oracle fields; `diagnostic-results.json` records FD2/4/6.

## Simulation command and restart provenance

`run-stress.sh` preserves the actual launch script, with fixed input/binary hash
assertions and a Gate 1 check. It invokes the established `cuda_driver.py run`
with `--wall-segment 00:15:00`. The five `segment-*.log.command.json` files preserve
every command, restart filename, working directory and launch time.
The first four segments ended cleanly on wall time, and the fifth failed strictly.
`stress-exit.json` records exit 1. No completed.json was emitted.

The head driver terminated; no new Slurm job ID exists. Baseline job ID is
13481403. The inspected partition was `gputest`, QoS `gpu-test`; its recorded
limits are historical evidence, not a substitute for checking limits before a
future allocated run. Do not relaunch into the finished evidence directories.

Gate 3 inputs in `inputs/convergence` are archived only. Gate 4 source input and
acceptance limitations are recorded in PLAN.md. Neither downstream run belongs
to the execution performed here.

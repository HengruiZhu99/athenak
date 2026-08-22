#!/usr/bin/env bash
set -euo pipefail

: "${CAMPAIGN_ROOT:?fresh campaign root required}"
: "${SOURCE_ROOT:?}"
: "${BUILD_ROOT:?}"

expected_source=5d37b5e5c278ac4a1afd52f9553dee6ffed48d0e
expected_tree=e391e2889647471e2e8c0cf8bfbfeb5fe3c00edf
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
expected_exe=6dadf42591f77bd9236a39230a8cf70290a68b61565dd0e9b6b624567691f54a
expected_cache=9809192574df929b2b70a550f48e1bea29d50369df62bbd65a5bb5e566fa4ec4

test ! -e "${CAMPAIGN_ROOT}"
case "${SLURM_JOB_QOS:-}" in
  shared_interactive|gpu_shared_interactive|gpu_shared_interactive_ss11) ;;
  *) printf 'unexpected QOS: %s\n' "${SLURM_JOB_QOS:-<missing>}" >&2; exit 2 ;;
esac
test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 2
test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_GPUS_PER_NODE:-}" = 1

test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD^{tree})" = "${expected_tree}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)"
test "$(git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"

athena="${BUILD_ROOT}/src/athena"
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${BUILD_ROOT}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"

profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
export COLLAPSE_ROOT="${CAMPAIGN_ROOT}"
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU

mkdir -p "${CAMPAIGN_ROOT}/evidence" "${CAMPAIGN_ROOT}/qualification"
root="$(cd "${CAMPAIGN_ROOT}" && pwd)"

finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s\n' "${code}" > "${root}/orchestration-status"
  git -C "${SOURCE_ROOT}" status --porcelain=v1 > "${root}/evidence/source-status.final"
  (cd "${root}" &&
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
      -print0 | LC_ALL=C sort -z | xargs -0r sha256sum > SHA256SUMS &&
    sha256sum --check SHA256SUMS >/dev/null &&
    sha256sum SHA256SUMS > SHA256SUMS.sha256)
  exit "${code}"
}
trap finish EXIT

env | LC_ALL=C sort > "${root}/evidence/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${root}/evidence/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${root}/evidence/node.txt"
{
  date --iso-8601=seconds
  git -C "${SOURCE_ROOT}" rev-parse HEAD HEAD^{tree}
  git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${BUILD_ROOT}/CMakeCache.txt"
  module list
} > "${root}/evidence/provenance.txt" 2>&1

regex='^athena[.]z4c_(vc|cartoon_axis_centered)'
ctest --test-dir "${BUILD_ROOT}" --show-only=json-v1 -R "${regex}" \
  > "${root}/qualification/ctest-inventory.json"
"${python_bin}" - "${root}/qualification/ctest-inventory.json" \
  "${BUILD_ROOT}" "${python_bin}" \
  > "${root}/qualification/cuda-ctest.log" 2>&1 <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time

inventory = json.loads(pathlib.Path(sys.argv[1]).read_text())
build_root = pathlib.Path(sys.argv[2])
python = sys.argv[3]
assert inventory["kind"] == "ctestInfo"
assert inventory["version"]["major"] == 1
tests = inventory["tests"]
assert len(tests) == 30, [test["name"] for test in tests]
assert len({test["name"] for test in tests}) == len(tests)
for index, test in enumerate(tests, 1):
    command = list(test["command"])
    if command[0] == "/usr/bin/python3":
        command[0] = python
    if "--mpiexec" not in command:
        command = [
            "/usr/bin/srun", "--nodes=1", "--ntasks=1",
            "--ntasks-per-node=1", "--cpus-per-task=8",
            "--gpus-per-task=1", "--gpu-bind=map_gpu:0",
            "--cpu-bind=cores", "--exact", "--kill-on-bad-exit=1",
        ] + command
    timeout = 300.0
    for prop in test.get("properties", []):
        if prop["name"] == "TIMEOUT":
            timeout = float(prop["value"])
        assert prop["name"] not in {
            "DISABLED", "WILL_FAIL", "SKIP_RETURN_CODE",
            "SKIP_REGULAR_EXPRESSION",
        }
    print(f"START {index}/30 {test['name']}", flush=True)
    start = time.monotonic()
    result = subprocess.run(
        command, cwd=build_root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, env=os.environ.copy())
    print(result.stdout, end="")
    print(
        f"END {index}/30 {test['name']} status={result.returncode} "
        f"seconds={time.monotonic()-start:.6f}", flush=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
print("100% tests passed, 0 tests failed out of 30", flush=True)
PY
grep -Fx '100% tests passed, 0 tests failed out of 30' \
  "${root}/qualification/cuda-ctest.log" >/dev/null
printf 'PERLMUTTER_FINAL_SOURCE_CUDA_VERTEX_TESTS_PASS\n' \
  > "${root}/qualification/status"

#!/bin/bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
campaign_root=$(cd "${bundle_dir}/.." && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NAME:-}" = "${EXPECTED_JOB_NAME}"
test "${SLURM_JOB_NUM_NODES:-}" = "${EXPECTED_NODES}"
test "${SLURM_NTASKS:-}" = "${EXPECTED_MPI_RANKS}"
test "${SLURM_GPUS_PER_NODE:-}" = "${EXPECTED_GPUS}"
test "${SLURM_CPUS_PER_TASK:-}" = "${EXPECTED_CPUS_PER_TASK}"
test -f "${campaign_root}/PREFLIGHT_COMPLETE"
test ! -e "${campaign_root}/run"
test ! -e "${campaign_root}/build"
(cd "${bundle_dir}" && sha256sum -c bundle.sha256 >/dev/null)

run_root=${campaign_root}/run
evidence=${run_root}/evidence
mkdir -p "${evidence}"
node_inventory=$(scontrol show hostnames "${SLURM_NODELIST:-}")
test "$(printf '%s\n' "${node_inventory}" | sed '/^$/d' | wc -l)" -eq 1
node=$(printf '%s\n' "${node_inventory}" | sed '/^$/d')
scontrol show node "${node}" > "${evidence}/node.scontrol.txt"
grep -Eq 'ActiveFeatures=[^ ]*gpu' "${evidence}/node.scontrol.txt"
grep -Eq 'ActiveFeatures=[^ ]*hbm40g' "${evidence}/node.scontrol.txt"
! grep -Eq 'ActiveFeatures=[^ ]*hbm80g' "${evidence}/node.scontrol.txt"

# shellcheck disable=SC1090
export COLLAPSE_ROOT="${campaign_root}"
source "${PROFILE_PATH}"
module load "${PYTHON_MODULE}"
test "$(sha256sum "${PROFILE_PATH}" | awk '{print $1}')" = \
  "${EXPECTED_PROFILE_SHA256}"
test "$(command -v python3)" = "${PYTHON_BIN}"
test "$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')" = \
  "${EXPECTED_PYTHON_SHA256}"
test "$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" = \
  "${EXPECTED_NUMPY_VERSION}"
test "$("${PYTHON_BIN}" -c 'import h5py; print(h5py.__version__)')" = \
  "${EXPECTED_H5PY_VERSION}"
test "$("${PYTHON_BIN}" -c 'import sympy; print(sympy.__version__)')" = \
  "${EXPECTED_SYMPY_VERSION}"
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD^{tree})" = "${EXPECTED_SOURCE_TREE}"
test -z "$(git -C "${SOURCE_ROOT}" status --short)"
test "$(git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD)" = "${EXPECTED_KOKKOS_COMMIT}"

export OMP_NUM_THREADS=${EXPECTED_CPUS_PER_TASK}
export KOKKOS_NUM_THREADS=${EXPECTED_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
export PYTHONDONTWRITEBYTECODE=1
env | sort > "${evidence}/environment.txt"
module -t list > "${evidence}/modules.txt" 2>&1
CC --version > "${evidence}/compiler.txt" 2>&1
cmake --version > "${evidence}/cmake.txt" 2>&1
nvcc --version > "${evidence}/nvcc.txt" 2>&1
git -C "${SOURCE_ROOT}" submodule status > "${evidence}/submodules.txt"

mkdir -p "${BUILD_ROOT}"
configure=(cmake -S "${SOURCE_ROOT}" -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=cc
  -DCMAKE_CXX_COMPILER=CC
  -DPROBLEM=built_in_pgens
  -DBUILD_TESTING=ON
  -DAthena_BUILD_UNIT_TESTS=ON
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}"
  -DAthena_SINGLE_PRECISION=OFF
  -DAthena_ENABLE_MPI=ON
  -DAthena_ENABLE_OPENMP=ON
  -DAthena_ENABLE_IRISK_INTERPOLATOR=OFF
  -DKokkos_ENABLE_TESTS=OFF
  -DKokkos_ENABLE_CUDA=ON
  -DKokkos_ENABLE_CUDA_LAMBDA=ON
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON
  -DKokkos_ENABLE_OPENMP=ON
  -DKokkos_ENABLE_SERIAL=OFF
  -DKokkos_ARCH_AMPERE80=ON)
printf '%q ' "${configure[@]}" > "${evidence}/configure-command.txt"
printf '\n' >> "${evidence}/configure-command.txt"
/usr/bin/time -p -o "${evidence}/configure-time.txt" \
  "${configure[@]}" > "${evidence}/configure.log" 2> "${evidence}/configure.err"

build=(cmake --build "${BUILD_ROOT}" --parallel 8)
printf '%q ' "${build[@]}" > "${evidence}/build-command.txt"
printf '\n' >> "${evidence}/build-command.txt"
/usr/bin/time -p -o "${evidence}/build-time.txt" \
  "${build[@]}" > "${evidence}/build.log" 2> "${evidence}/build.err"
test -x "${EXECUTABLE}"
sha256sum "${EXECUTABLE}" "${BUILD_ROOT}/CMakeCache.txt" > \
  "${evidence}/build-products.sha256"
"${EXECUTABLE}" -c > "${evidence}/athena-config.txt" 2> \
  "${evidence}/athena-config.err"
grep -F 'MPI parallelism:            ON' "${evidence}/athena-config.txt" >/dev/null
grep -F 'AthenaK Kokkos default execution space: Cuda' \
  "${evidence}/athena-config.txt" >/dev/null
grep -Fx 'Athena_BUILD_UNIT_TESTS:BOOL=ON' "${BUILD_ROOT}/CMakeCache.txt" >/dev/null
grep -Fx 'Athena_ENABLE_IRISK_INTERPOLATOR:BOOL=OFF' \
  "${BUILD_ROOT}/CMakeCache.txt" >/dev/null

ctest --test-dir "${BUILD_ROOT}" -N > "${evidence}/ctest-inventory.txt"
/usr/bin/time -p -o "${evidence}/ctest-time.txt" \
  ctest --test-dir "${BUILD_ROOT}" --output-on-failure -j 1 > \
  "${evidence}/ctest.log" 2> "${evidence}/ctest.err"
grep -F 'athena.z4c_cartoon_production_kernels_cuda_required' \
  "${evidence}/ctest-inventory.txt" >/dev/null
grep -F 'athena.pdf_scatter_cuda_required' "${evidence}/ctest-inventory.txt" >/dev/null
! grep -Fq '(Disabled)' "${evidence}/ctest-inventory.txt"

driver=${SOURCE_ROOT}/tst/test_suite/z4c/cartoon_half_plane_kerr_campaign.py
analyzer=${SOURCE_ROOT}/tst/test_suite/z4c/cartoon_half_plane_kerr_convergence.py
template=${SOURCE_ROOT}/tst/inputs/z4c_kerr_half_plane_convergence.athinput
rank_wrapper=${SOURCE_ROOT}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
campaign=${run_root}/campaign
"${PYTHON_BIN}" -B "${driver}" prepare --source-dir "${SOURCE_ROOT}" \
  --executable "${EXECUTABLE}" --template "${template}" \
  --analyzer "${analyzer}" --output "${campaign}"

for case_name in moving_puncture_h32 moving_puncture_h48 moving_puncture_h64; do
  spacing=${case_name##*_}
  bindings=${campaign}/moving_puncture/${spacing}/bindings
  launcher_json=$("${PYTHON_BIN}" -B - "${rank_wrapper}" "${bindings}" <<'PY'
import json, sys
print(json.dumps([
    "srun", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
    "--cpus-per-task=8", "--gpus-per-task=1",
    "--gpu-bind=map_gpu:0,1,2,3", "--cpu-bind=cores", "--exact",
    "--kill-on-bad-exit=1", sys.executable, sys.argv[1],
    "--evidence-dir", sys.argv[2], "--require-cuda", "--"
]))
PY
)
  "${PYTHON_BIN}" -B "${driver}" run-case --output "${campaign}" \
    --case "${case_name}" --launcher-json "${launcher_json}"
done

"${PYTHON_BIN}" -B "${driver}" analyze --output "${campaign}"
"${PYTHON_BIN}" -B - "${campaign}" "${EXPECTED_SOURCE_COMMIT}" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
def reject(token): raise ValueError(f"nonfinite JSON token {token}")
def load(path):
    return json.loads(path.read_text(), parse_constant=reject,
                      object_pairs_hook=lambda pairs: _strict(pairs))
def _strict(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError(f"duplicate key {key}")
        out[key] = value
    return out
state = load(root / "campaign_state.json")
analysis = load(root / "qualification_analysis.json")
assert state["contract"]["source"]["commit"] == sys.argv[2]
assert list(state["cases"]) == ["moving_puncture_h32", "moving_puncture_h48",
                                "moving_puncture_h64"]
assert all(record["status"] == "complete" for record in state["cases"].values())
uuids = set()
for spacing in ("h32", "h48", "h64"):
    bindings = sorted((root / "moving_puncture" / spacing / "bindings").glob(
        "rank_binding_*.json"))
    assert len(bindings) == 4
    records = [load(path) for path in bindings]
    assert sorted(record["rank"] for record in records) == [0, 1, 2, 3]
    assert all(record["binding_verified"] is True for record in records)
    assert all(record["gpu_name"] == "NVIDIA A100-SXM4-40GB" for record in records)
    current = {record["selected_uuid"] for record in records}
    assert len(current) == 4
    if uuids: assert current == uuids
    uuids = current
gauge = state["contract"]["physics"]["gauge"]
assert gauge == {
    "name": "athenak_default_moving_puncture",
    "lapse": "advective_1_plus_log", "lapse_oplog": 2.0,
    "lapse_harmonicf": 1.0, "lapse_harmonic": 0.0,
    "lapse_advect": 1.0, "slow_start_lapse": False,
    "telegraph_lapse": False, "shift": "advective_Gamma_driver",
    "shift_Gamma": 1.0, "shift_eta": 2.0, "shift_advect": 1.0,
    "shift_alpha2Gamma": 0.0, "shift_H": 0.0,
    "shift_eta_max_K": False, "sss_damping_amp": 0.0}
assert analysis["qualification_claim"] is True
assert analysis["gauges"]["moving_puncture"]["gates"]["gauge_qualification"] is True
PY
printf 'HALF_PLANE_KERR_MOVING_PUNCTURE_PASS\n' > "${run_root}/verdict.txt"

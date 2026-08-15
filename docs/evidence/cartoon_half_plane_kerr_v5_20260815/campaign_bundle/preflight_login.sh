#!/bin/bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
campaign_root=$(cd "${bundle_dir}/.." && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
test -z "${SLURM_JOB_ID:-}"
test ! -e "${campaign_root}/PREFLIGHT_COMPLETE"
test ! -e "${campaign_root}/preflight.sha256"
test ! -e "${campaign_root}/build"
test ! -e "${campaign_root}/run"
test ! -e "${campaign_root}/allocation"
test -z "$(squeue -h -u "${USER}" -n "${EXPECTED_JOB_NAME}" -o '%A')"
(cd "${bundle_dir}" && sha256sum -c bundle.sha256 >/dev/null)
test -z "$(find "${bundle_dir}" \( -type d -name __pycache__ -o \
  -type f -name '*.pyc' \) -print -quit)"
test "$(sha256sum "${PROFILE_PATH}" | awk '{print $1}')" = \
  "${EXPECTED_PROFILE_SHA256}"
module load "${PYTHON_MODULE}"
test "$(command -v python3)" = "${PYTHON_BIN}"
test -x "${PYTHON_BIN}"
test "$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')" = \
  "${EXPECTED_PYTHON_SHA256}"
test "$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[:2] >= (3, 11))')" = True
test "$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" = \
  "${EXPECTED_NUMPY_VERSION}"
test "$("${PYTHON_BIN}" -c 'import h5py; print(h5py.__version__)')" = \
  "${EXPECTED_H5PY_VERSION}"
test "$("${PYTHON_BIN}" -c 'import sympy; print(sympy.__version__)')" = \
  "${EXPECTED_SYMPY_VERSION}"
test -d "${V1_INCOMPLETE_ROOT}"
test ! -e "${V1_FINAL_ROOT}"
test ! -e "${V1_INCOMPLETE_ROOT}/build"
test ! -e "${V1_INCOMPLETE_ROOT}/run"
test ! -e "${V1_INCOMPLETE_ROOT}/allocation"
test ! -e "${V1_INCOMPLETE_ROOT}/PREFLIGHT_COMPLETE"
test "$(sha256sum "${V1_INCOMPLETE_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V1_BUNDLE_MANIFEST_SHA256}"
(cd "${V1_INCOMPLETE_ROOT}/bundle" && sha256sum -c bundle.sha256 >/dev/null)
test "$(git -C "${V1_INCOMPLETE_ROOT}/source/athenak" rev-parse HEAD)" = \
  "${EXPECTED_PREDECESSOR_SOURCE_COMMIT}"
test "$(git -C "${V1_INCOMPLETE_ROOT}/source/athenak" rev-parse HEAD^{tree})" = \
  "${EXPECTED_PREDECESSOR_SOURCE_TREE}"
test -z "$(git -C "${V1_INCOMPLETE_ROOT}/source/athenak" status --short)"
test -d "${V2_INCOMPLETE_ROOT}"
test ! -e "${V2_FINAL_ROOT}"
test ! -e "${V2_INCOMPLETE_ROOT}/build"
test ! -e "${V2_INCOMPLETE_ROOT}/run"
test ! -e "${V2_INCOMPLETE_ROOT}/allocation"
test ! -e "${V2_INCOMPLETE_ROOT}/PREFLIGHT_COMPLETE"
test "$(sha256sum "${V2_INCOMPLETE_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V2_BUNDLE_MANIFEST_SHA256}"
(cd "${V2_INCOMPLETE_ROOT}/bundle" && sha256sum -c bundle.sha256 >/dev/null)
test "$(git -C "${V2_INCOMPLETE_ROOT}/source/athenak" rev-parse HEAD)" = \
  "${EXPECTED_PREDECESSOR_SOURCE_COMMIT}"
test "$(git -C "${V2_INCOMPLETE_ROOT}/source/athenak" rev-parse HEAD^{tree})" = \
  "${EXPECTED_PREDECESSOR_SOURCE_TREE}"
test -z "$(git -C "${V2_INCOMPLETE_ROOT}/source/athenak" status --short)"
test "$(sha256sum "${V3_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_V3_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${V3_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V3_DETACHED_SHA256}"
test "$(sha256sum "${V3_ROOT}/allocation/sacct-settled.psv" | awk '{print $1}')" = \
  "${EXPECTED_V3_SACCT_SHA256}"
test "$(sha256sum "${V3_ROOT}/allocation/source-status-final.txt" | awk '{print $1}')" = \
  "${EXPECTED_V3_SOURCE_STATUS_SHA256}"
test "$(sha256sum "${V3_ROOT}/run/evidence/build.err" | awk '{print $1}')" = \
  "${EXPECTED_V3_BUILD_ERR_SHA256}"
(cd "${V3_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null && \
  sha256sum -c SHA256SUMS >/dev/null)
test "$(cat "${V3_ROOT}/allocation/job-id.txt")" = "${V3_JOB_ID}"
test "$(cat "${V3_ROOT}/allocation/accounting-verdict.txt")" = \
  'mode=failure numbered_steps=0'
test "$(cat "${V3_ROOT}/allocation/allocation.status")" = 2
test ! -e "${V3_ROOT}/run/campaign"
test "$(git -C "${V3_ROOT}/source/athenak" rev-parse HEAD)" = \
  "${EXPECTED_PREDECESSOR_SOURCE_COMMIT}"
test "$(git -C "${V3_ROOT}/source/athenak" rev-parse HEAD^{tree})" = \
  "${EXPECTED_PREDECESSOR_SOURCE_TREE}"
test -z "$(git -C "${V3_ROOT}/source/athenak" status --short)"
test "$(sha256sum "${V4_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_V4_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${V4_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V4_DETACHED_SHA256}"
test "$(sha256sum "${V4_ROOT}/allocation/sacct-settled.psv" | awk '{print $1}')" = \
  "${EXPECTED_V4_SACCT_SHA256}"
test "$(sha256sum "${V4_ROOT}/allocation/source-status-final.txt" | awk '{print $1}')" = \
  "${EXPECTED_V4_SOURCE_STATUS_SHA256}"
test "$(sha256sum "${V4_ROOT}/run/evidence/ctest.log" | awk '{print $1}')" = \
  "${EXPECTED_V4_CTEST_LOG_SHA256}"
test "$(sha256sum "${V4_ROOT}/run/evidence/ctest.err" | awk '{print $1}')" = \
  "${EXPECTED_V4_CTEST_ERR_SHA256}"
test "$(sha256sum "${V4_ROOT}/build/athena-cuda/src/athena" | awk '{print $1}')" = \
  "${EXPECTED_V4_EXECUTABLE_SHA256}"
(cd "${V4_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null && \
  sha256sum -c SHA256SUMS >/dev/null)
test "$(cat "${V4_ROOT}/allocation/job-id.txt")" = "${V4_JOB_ID}"
test "$(cat "${V4_ROOT}/allocation/accounting-verdict.txt")" = \
  'mode=failure numbered_steps=0'
test "$(cat "${V4_ROOT}/allocation/allocation.status")" = 8
test ! -e "${V4_ROOT}/run/campaign"
grep -Fq '92% tests passed, 3 tests failed out of 37' \
  "${V4_ROOT}/run/evidence/ctest.log"
grep -Fq 'athena.z4c_cartoon_mms_static (Failed)' \
  "${V4_ROOT}/run/evidence/ctest.log"
grep -Fq 'athena.z4c_cartoon_axis_boundary (Subprocess aborted)' \
  "${V4_ROOT}/run/evidence/ctest.log"
grep -Fq 'athena.z4c_kerr_half_plane_init (Failed)' \
  "${V4_ROOT}/run/evidence/ctest.log"
grep -Fq 'Cartoon MMS oracle requires SymPy 1.14.0, got 1.12' \
  "${V4_ROOT}/run/evidence/ctest.log"
grep -Fq 'cudaErrorIllegalInstruction' "${V4_ROOT}/run/evidence/ctest.log"
grep -Fq 'numpy.dtype size changed' "${V4_ROOT}/run/evidence/ctest.log"
test "$(git -C "${V4_ROOT}/source/athenak" rev-parse HEAD)" = \
  "${EXPECTED_AXIS_TEST_REPAIR_PARENT}"
test -z "$(git -C "${V4_ROOT}/source/athenak" status --short)"

source_root=${campaign_root}/source/athenak
test ! -e "${source_root}"
mkdir -p "${campaign_root}/source"
git clone -q --no-checkout "${ATHENA_URL}" "${source_root}"
git -C "${source_root}" fetch -q origin \
  "${ATHENA_BRANCH}:refs/remotes/origin/${ATHENA_BRANCH}"
test "$(git -C "${source_root}" rev-parse \
  "refs/remotes/origin/${ATHENA_BRANCH}")" = "${EXPECTED_SOURCE_COMMIT}"
git -C "${source_root}" checkout -q --detach "${EXPECTED_SOURCE_COMMIT}"
git -C "${source_root}" submodule update -q --init --recursive
test "$(git -C "${source_root}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test "$(git -C "${source_root}" rev-parse HEAD^{tree})" = "${EXPECTED_SOURCE_TREE}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = \
  "${EXPECTED_KOKKOS_COMMIT}"
test -z "$(git -C "${source_root}" status --short)"
test "$(git -C "${source_root}" rev-parse HEAD^)" = \
  "${EXPECTED_AXIS_TEST_REPAIR_PARENT}"
test "$(git -C "${source_root}" show HEAD | git patch-id --stable | awk '{print $1}')" = \
  "${EXPECTED_AXIS_TEST_REPAIR_PATCH_ID}"
test "$(git -C "${source_root}" diff --name-only HEAD^..HEAD | paste -sd '|' -)" = \
  "${EXPECTED_AXIS_TEST_REPAIR_PATH}"
test "$(sha256sum "${source_root}/${EXPECTED_AXIS_TEST_REPAIR_PATH}" | awk '{print $1}')" = \
  "${EXPECTED_AXIS_TEST_SHA256}"
test "$(git -C "${source_root}" rev-parse HEAD^^)" = \
  "${EXPECTED_KERR_REPAIR_PARENT}"
test "$(git -C "${source_root}" show HEAD^ | git patch-id --stable | awk '{print $1}')" = \
  "${EXPECTED_KERR_REPAIR_PATCH_ID}"
test "$(git -C "${source_root}" diff --name-only HEAD^^..HEAD^ | paste -sd '|' -)" = \
  "${EXPECTED_KERR_REPAIR_PATHS}"
test "$(sha256sum "${source_root}/src/pgen/z4c/kerr_puncture.cpp" | awk '{print $1}')" = \
  "${EXPECTED_KERR_PGEN_SHA256}"
test "$(sha256sum "${source_root}/tst/unit/z4c/z4c_kerr_half_plane_static_test.py" | awk '{print $1}')" = \
  "${EXPECTED_KERR_STATIC_SHA256}"

template=${source_root}/tst/inputs/z4c_kerr_half_plane_convergence.athinput
driver=${source_root}/tst/test_suite/z4c/cartoon_half_plane_kerr_campaign.py
analyzer=${source_root}/tst/test_suite/z4c/cartoon_half_plane_kerr_convergence.py
rank_wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
test "$(sha256sum "${template}" | awk '{print $1}')" = \
  "${EXPECTED_TEMPLATE_SHA256}"
test "$(sha256sum "${driver}" | awk '{print $1}')" = "${EXPECTED_DRIVER_SHA256}"
test "$(sha256sum "${analyzer}" | awk '{print $1}')" = \
  "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${rank_wrapper}" | awk '{print $1}')" = \
  "${EXPECTED_RANK_WRAPPER_SHA256}"
test "$(sha256sum "${source_root}/CMakeLists.txt" | awk '{print $1}')" = \
  "${EXPECTED_CMAKE_SHA256}"
test "$(sha256sum "${source_root}/docs/z4c_cartoon_half_plane_operator_table.md" | \
  awk '{print $1}')" = "${EXPECTED_OPERATOR_TABLE_SHA256}"

"${PYTHON_BIN}" -B "${driver}" self-test --template "${template}"
"${PYTHON_BIN}" -B "${analyzer}" --self-test
"${PYTHON_BIN}" -B \
  "${source_root}/tst/unit/z4c/generate_z4c_cartoon_operator_table.py" \
  --source-dir "${source_root}"
for token in \
  'lapse_oplog = 2.0' 'lapse_harmonicf = 1.0' \
  'lapse_harmonic = 0.0' 'lapse_advect = 1.0' \
  'slow_start_lapse = false' 'telegraph_lapse = false' \
  'shift_Gamma = 1.0' 'shift_eta = 2.0' 'shift_advect = 1.0' \
  'shift_alpha2Gamma = 0.0' 'shift_H = 0.0' \
  'shift_eta_max_K = false' 'sss_damping_amp = 0.0' \
  'initial_gauge = precollapsed' 'dchi_max = 0.02' 'tlim = 5.0'; do
  test "$(grep -Fxc "${token}" "${template}")" -eq 1
done
grep -Fq 'GetOrAddReal("z4c", "lapse_oplog", 2.0)' \
  "${source_root}/src/z4c/z4c.cpp"
grep -Fq 'GetOrAddReal("z4c", "shift_eta", 2.0)' \
  "${source_root}/src/z4c/z4c.cpp"
grep -Fq 'GetOrAddBoolean("z4c", "telegraph_lapse", false)' \
  "${source_root}/src/z4c/z4c.cpp"

"${PYTHON_BIN}" -B - "${campaign_root}/source-preflight.json" \
  "${SOURCE_ROOT}" "${EXPECTED_SOURCE_COMMIT}" "${EXPECTED_SOURCE_TREE}" \
  "${EXPECTED_KOKKOS_COMMIT}" "${EXPECTED_JOB_NAME}" <<'PY'
import json, pathlib, sys
payload = {
    "schema": "athenak_cartoon_half_plane_kerr_preflight_v1",
    "declared_promoted_source_path": sys.argv[2],
    "source_commit": sys.argv[3], "source_tree": sys.argv[4],
    "kokkos_commit": sys.argv[5], "job_name": sys.argv[6],
    "case_inventory": ["moving_puncture_h32", "moving_puncture_h48",
                       "moving_puncture_h64"],
    "gauge": "default_advective_1_plus_log_and_Gamma_driver",
    "qualification_claim": False,
    "allocation_launched": False,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2,
                                                sort_keys=True) + "\n")
PY
printf 'NO_ALLOCATION_PREFLIGHT_PASS\n' > "${campaign_root}/PREFLIGHT_COMPLETE"
(
  cd "${campaign_root}"
  sha256sum bundle/bundle.sha256 source-preflight.json PREFLIGHT_COMPLETE > \
    preflight.sha256
  sha256sum -c preflight.sha256 >/dev/null
)

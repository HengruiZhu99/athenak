#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${1:-$(cd "${bundle_dir}/.." && pwd)}
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
test "${root}" = "${REMOTE_ROOT}.incomplete"
test -d "${root}/bundle"
test ! -e "${REMOTE_ROOT}"
test ! -e "${root}/build"
test ! -e "${root}/run"
test ! -e "${root}/allocation"
test -z "$(squeue -h -u "${USER}" -n "${EXPECTED_JOB_NAME}" -o '%A')"
(cd "${bundle_dir}" && sha256sum -c bundle.sha256)

test "$(sha256sum "${PROFILE_PATH}" | awk '{print $1}')" = "${EXPECTED_PROFILE_SHA256}"
test "$(sha256sum "${bundle_dir}/${INPUT_BASENAME}" | awk '{print $1}')" = \
  "${EXPECTED_INPUT_SHA256}"
test "$(sha256sum "${bundle_dir}/${COEFFICIENT_BASENAME}" | awk '{print $1}')" = \
  "${EXPECTED_COEFFICIENT_SHA256}"
test "$(git -C "${SOURCE_PATH}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test "$(git -C "${SOURCE_PATH}" rev-parse HEAD^{tree})" = "${EXPECTED_SOURCE_TREE}"
test "$(git -C "${SOURCE_PATH}/kokkos" rev-parse HEAD)" = "${EXPECTED_SOURCE_KOKKOS}"
test -z "$(git -C "${SOURCE_PATH}" status --short)"
test "$(sha256sum "${EXECUTABLE}" | awk '{print $1}')" = "${EXPECTED_EXECUTABLE_SHA256}"

test "$(sha256sum "${BASELINE_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_BASELINE_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${BASELINE_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_BASELINE_DETACHED_SHA256}"
(cd "${BASELINE_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null)
test "$(sha256sum "${BASELINE_CASE}/brill_fig3_max_domain_abs_K_n128.z4c.user.hst" | awk '{print $1}')" = \
  "${EXPECTED_BASELINE_HISTORY_SHA256}"

test "$(sha256sum "${PRIOR_FIXED_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${PRIOR_FIXED_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_DETACHED_SHA256}"
(cd "${PRIOR_FIXED_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null)
test "$(sha256sum "${PRIOR_FIXED_CASE}/brill_fig3_fixed_eta2_n128.z4c.user.hst" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_HISTORY_SHA256}"
test "$(sha256sum "${PRIOR_FIXED_CASE}/run.log" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_LOG_SHA256}"
test "$(sha256sum "${PRIOR_FIXED_CASE}/command.txt" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_COMMAND_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_BUNDLE_MANIFEST_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/preflight.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_PREFLIGHT_MANIFEST_SHA256}"
test "$(sha256sum "${bundle_dir}/v1-failure.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V1_LEDGER_SHA256}"
(cd "${V1_ROOT}" && sha256sum -c "${bundle_dir}/v1-failure.sha256" >/dev/null)
test ! -e "${V1_ROOT}/run"
grep -Fq "${EXPECTED_V1_JOB_ID}|${EXPECTED_V1_JOB_NAME}|CANCELLED by " \
  "${V1_ROOT}/allocation/sacct-settled.psv"
test -d "${V3_INCOMPLETE_ROOT}"
test ! -e "${V3_INCOMPLETE_ROOT%.incomplete}"
test ! -e "${V3_INCOMPLETE_ROOT}/run"
test ! -e "${V3_INCOMPLETE_ROOT}/allocation"
test "$(sha256sum "${V3_INCOMPLETE_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V3_BUNDLE_MANIFEST_SHA256}"
(cd "${V3_INCOMPLETE_ROOT}/bundle" && sha256sum -c bundle.sha256 >/dev/null)
test -d "${V4_ROOT}"
test ! -e "${V4_ROOT}/run"
test ! -e "${V4_ROOT}/allocation"
test "$(sha256sum "${V4_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V4_BUNDLE_MANIFEST_SHA256}"
test "$(sha256sum "${V4_ROOT}/preflight.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V4_PREFLIGHT_MANIFEST_SHA256}"
(cd "${V4_ROOT}/bundle" && sha256sum -c bundle.sha256 >/dev/null)
test -d "${V5_ROOT}"
test ! -e "${V5_ROOT}/run"
test "$(sha256sum "${V5_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V5_BUNDLE_MANIFEST_SHA256}"
test "$(sha256sum "${V5_ROOT}/preflight.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V5_PREFLIGHT_MANIFEST_SHA256}"
test "$(sha256sum "${V5_ROOT}/allocation/predecessor-terminal.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V5_EMPTY_PREDECESSOR_LOG_SHA256}"
(cd "${V5_ROOT}/bundle" && sha256sum -c bundle.sha256 >/dev/null)
test -z "$(squeue -h -u "${USER}" -n cartoon-r4-domain64-nocd-ko05-v5 -o '%A')"
test "$(sha256sum "${V6_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_V6_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${V6_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_V6_DETACHED_SHA256}"
test "$(sha256sum "${V6_ROOT}/allocation/sacct-settled.psv" | awk '{print $1}')" = \
  "${EXPECTED_V6_SACCT_SHA256}"
test "$(sha256sum "${V6_ROOT}/run/comparison.json" | awk '{print $1}')" = \
  "${EXPECTED_V6_COMPARISON_SHA256}"
test "$(sha256sum "${V6_ROOT}/run/cases/${CASE_KO05}/run.log" | awk '{print $1}')" = \
  "${EXPECTED_V6_FIXED_LOG_SHA256}"
test "$(sha256sum "${V6_ROOT}/run/cases/${CASE_ZERO_SHIFT_KO05}/run.log" | awk '{print $1}')" = \
  "${EXPECTED_V6_ZERO_LOG_SHA256}"
(cd "${V6_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null && \
  sha256sum -c SHA256SUMS >/dev/null)
test -z "$(squeue -h -u "${USER}" -n "${EXPECTED_V6_JOB_NAME}" -o '%A')"
"${PYTHON_BIN}" -B - "${V6_ROOT}/allocation/sacct-settled.psv" \
  "${V6_ROOT}/run/comparison.json" "${EXPECTED_V6_JOB_ID}" \
  "${EXPECTED_V6_JOB_NAME}" <<'PY'
import json, pathlib, sys
sacct, comparison, job_id, job_name=sys.argv[1:]
rows=[line.split('|') for line in pathlib.Path(sacct).read_text().splitlines() if line]
assert [r[0] for r in rows if r[0] == job_id] == [job_id]
parent=[r for r in rows if r[0] == job_id][0]
assert parent[1:4] == [job_name, 'FAILED', '1:0']
steps=[r for r in rows if r[0] in {job_id+'.0', job_id+'.1'}]
assert [r[0] for r in steps] == [job_id+'.0', job_id+'.1']
assert all(r[2] == 'CANCELLED' and r[3] == '0:6' for r in steps)
d=json.loads(pathlib.Path(comparison).read_text())
assert d['all_cases_attempted'] is True
assert d['qualification_claim'] is False
assert len(d['cases']) == 2
for case in d['cases']:
    assert case['exit_code'] == 134
    assert case['history_rows'] == 0
    assert case['fatal_lines'] == [
        'what():  Kokkos ERROR: Cuda memory space failed to allocate 4.883 GiB (label="lb send data").'
    ]
    assert len(case['rank_bindings']) == 1
    assert case['rank_bindings'][0]['gpu_name'] == 'NVIDIA A100-SXM4-40GB'
PY
mapfile -t predecessor_queue < <(squeue -h -j "${EXPECTED_PREDECESSOR_JOB_ID}" \
  -o '%i|%j|%T')
case "${#predecessor_queue[@]}" in
  0)
    bash "${bundle_dir}/verify_predecessor.sh" >/dev/null
    ;;
  1)
    case "${predecessor_queue[0]}" in
      "${EXPECTED_PREDECESSOR_JOB_ID}|${EXPECTED_PREDECESSOR_JOB_NAME}|RUNNING"|\
      "${EXPECTED_PREDECESSOR_JOB_ID}|${EXPECTED_PREDECESSOR_JOB_NAME}|PENDING") ;;
      *) printf 'unexpected predecessor queue record: %s\n' "${predecessor_queue[0]}" >&2; exit 2 ;;
    esac
    ;;
  *) printf 'duplicate predecessor queue records\n' >&2; exit 2 ;;
esac

bash "${bundle_dir}/self_test.sh"
mkdir "${root}/preflight"
git -C "${SOURCE_PATH}" status --porcelain=v1 > "${root}/preflight/source-status.txt"
sha256sum "${bundle_dir}/bundle.sha256" "${EXECUTABLE}" \
  "${BASELINE_ROOT}/SHA256SUMS" "${BASELINE_ROOT}/SHA256SUMS.sha256" \
  "${PRIOR_FIXED_ROOT}/SHA256SUMS" "${PRIOR_FIXED_ROOT}/SHA256SUMS.sha256" \
  "${PREDECESSOR_ROOT}/bundle/bundle.sha256" "${PREDECESSOR_ROOT}/preflight.sha256" \
  "${bundle_dir}/predecessor-selected.sha256" \
  "${V6_ROOT}/SHA256SUMS" "${V6_ROOT}/SHA256SUMS.sha256" \
  "${V6_ROOT}/allocation/sacct-settled.psv" "${V6_ROOT}/run/comparison.json" > \
  "${root}/preflight/prerequisites.sha256"
sha256sum "${bundle_dir}/v1-failure.sha256" \
  "${V1_ROOT}/allocation/sacct-settled.psv" \
  "${V3_INCOMPLETE_ROOT}/bundle/bundle.sha256" \
  "${V4_ROOT}/bundle/bundle.sha256" "${V4_ROOT}/preflight.sha256" \
  "${V5_ROOT}/bundle/bundle.sha256" "${V5_ROOT}/preflight.sha256" \
  "${V5_ROOT}/allocation/predecessor-terminal.sha256" > \
  "${root}/preflight/v1-failure.sha256"
printf '%s\n' NO_ALLOCATION_PREFLIGHT_PASS > "${root}/PREFLIGHT_COMPLETE"
(cd "${root}" && find bundle preflight -type f -print0 | sort -z | \
  xargs -0 sha256sum) > "${root}/preflight.sha256"

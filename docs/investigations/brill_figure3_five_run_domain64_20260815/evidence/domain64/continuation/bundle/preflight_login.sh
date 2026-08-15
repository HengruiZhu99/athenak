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
test ! -e "${root}/run"
test ! -e "${root}/allocation"
test -z "$(squeue -h -u "${USER}" -n "${EXPECTED_JOB_NAME}" -o '%A')"
(cd "${bundle_dir}" && sha256sum -c bundle.sha256)
test "$(sha256sum "${PROFILE_PATH}" | awk '{print $1}')" = "${EXPECTED_PROFILE_SHA256}"
test "$(sha256sum "${bundle_dir}/brill_global_48x32.coefficients" | awk '{print $1}')" = \
  "${EXPECTED_COEFFICIENT_SHA256}"
test "$(git -C "${SOURCE_PATH}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test "$(git -C "${SOURCE_PATH}" rev-parse HEAD^{tree})" = "${EXPECTED_SOURCE_TREE}"
test "$(git -C "${SOURCE_PATH}/kokkos" rev-parse HEAD)" = "${EXPECTED_SOURCE_KOKKOS}"
test -z "$(git -C "${SOURCE_PATH}" status --short)"
test "$(sha256sum "${EXECUTABLE}" | awk '{print $1}')" = "${EXPECTED_EXECUTABLE_SHA256}"
test "$(sha256sum "${V7_ROOT}/SHA256SUMS" | awk '{print $1}')" = "${EXPECTED_V7_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${V7_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = "${EXPECTED_V7_DETACHED_SHA256}"
test "$(sha256sum "${V7_ROOT}/allocation/sacct-settled.psv" | awk '{print $1}')" = "${EXPECTED_V7_SACCT_SHA256}"
test "$(sha256sum "${V7_ROOT}/run/comparison.json" | awk '{print $1}')" = "${EXPECTED_V7_COMPARISON_SHA256}"
mapfile -t histories < <(find "${V7_ROOT}/run/cases/${V7_ZERO_CASE}" \
  -maxdepth 1 -type f -name '*.hst' -print)
test "${#histories[@]}" -eq 1 && test ! -L "${histories[0]}"
history=${histories[0]}
test "$(sha256sum "${history}" | awk '{print $1}')" = "${EXPECTED_V7_ZERO_HISTORY_SHA256}"
test "$(sha256sum "${V7_ROOT}/run/cases/${V7_ZERO_CASE}/run.log" | awk '{print $1}')" = "${EXPECTED_V7_ZERO_LOG_SHA256}"
restart=${V7_ROOT}/run/cases/${V7_ZERO_CASE}/rst/${RESTART_BASENAME}
test -f "${restart}" && test ! -L "${restart}"
test "$(stat -c '%s' "${restart}")" = "${EXPECTED_RESTART_BYTES}"
test "$(sha256sum "${restart}" | awk '{print $1}')" = "${EXPECTED_RESTART_SHA256}"
"${PYTHON_BIN}" -B - "${V7_ROOT}/allocation/sacct-settled.psv" \
  "${V7_ROOT}/run/comparison.json" "${EXPECTED_V7_JOB_ID}" \
  "${EXPECTED_V7_JOB_NAME}" "${V7_ZERO_CASE}" <<'PY'
import json, pathlib, sys
sacct, comparison, jid, jname, zero = sys.argv[1:]
rows=[line.split('|') for line in pathlib.Path(sacct).read_text().splitlines() if line]
parent=[r for r in rows if r[0] == jid]
assert len(parent) == 1 and parent[0][1:4] == [jname, 'FAILED', '1:0']
steps=[r for r in rows if r[0] in {jid+'.0', jid+'.1'}]
assert [r[0] for r in steps] == [jid+'.0', jid+'.1']
assert steps[0][2:4] == ['FAILED', '1:0']
assert steps[1][2:4] == ['TIMEOUT', '0:15']
d=json.loads(pathlib.Path(comparison).read_text())
assert d['all_cases_attempted'] is True and d['qualification_claim'] is False
assert len(d['cases']) == 2
z=[c for c in d['cases'] if c['name'] == zero]
assert len(z) == 1
z=z[0]
assert z['exit_code'] == 143 and z['reached_target_t20'] is False
assert z['history_rows'] == 5248
assert z['last_history']['time'] == 7.42498168945451
assert z['last_history']['cycle'] == 20988.0
assert z['last_history']['maxRefLev'] == 11.0
assert z['fatal_lines'] == [
  '[2026-08-15T09:32:43.248] error: *** STEP 57004715.1 ON nid008512 CANCELLED AT 2026-08-15T09:32:43 DUE TO TIME LIMIT ***'
]
PY
bash "${bundle_dir}/self_test.sh"
mkdir "${root}/preflight"
git -C "${SOURCE_PATH}" status --porcelain=v1 > "${root}/preflight/source-status.txt"
sha256sum "${bundle_dir}/bundle.sha256" "${EXECUTABLE}" \
  "${V7_ROOT}/SHA256SUMS" "${V7_ROOT}/SHA256SUMS.sha256" \
  "${V7_ROOT}/allocation/sacct-settled.psv" "${V7_ROOT}/run/comparison.json" \
  "${history}" "${V7_ROOT}/run/cases/${V7_ZERO_CASE}/run.log" "${restart}" > \
  "${root}/preflight/prerequisites.sha256"
printf '%s\n' NO_ALLOCATION_PREFLIGHT_PASS > "${root}/PREFLIGHT_COMPLETE"
(cd "${root}" && find bundle preflight -type f -print0 | sort -z | \
  xargs -0 sha256sum) > "${root}/preflight.sha256"

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
bash "${bundle_dir}/verify_predecessor.sh"

bash "${bundle_dir}/self_test.sh"
mkdir "${root}/preflight"
git -C "${SOURCE_PATH}" status --porcelain=v1 > "${root}/preflight/source-status.txt"
sha256sum "${bundle_dir}/bundle.sha256" "${EXECUTABLE}" \
  "${BASELINE_ROOT}/SHA256SUMS" "${BASELINE_ROOT}/SHA256SUMS.sha256" \
  "${PRIOR_FIXED_ROOT}/SHA256SUMS" "${PRIOR_FIXED_ROOT}/SHA256SUMS.sha256" \
  "${PREDECESSOR_ROOT}/SHA256SUMS" "${PREDECESSOR_ROOT}/SHA256SUMS.sha256" > \
  "${root}/preflight/prerequisites.sha256"
printf '%s\n' NO_ALLOCATION_PREFLIGHT_PASS > "${root}/PREFLIGHT_COMPLETE"
(cd "${root}" && find bundle preflight -type f -print0 | sort -z | \
  xargs -0 sha256sum) > "${root}/preflight.sha256"

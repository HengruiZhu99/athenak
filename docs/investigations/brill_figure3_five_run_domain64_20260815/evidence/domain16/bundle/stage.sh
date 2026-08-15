#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
bash "${bundle_dir}/self_test.sh"
host=hzhu@perlmutter.nersc.gov
ssh_opts=(-i "${HOME}/.ssh/nersc" -o IdentitiesOnly=yes -o BatchMode=yes)
incomplete=${REMOTE_ROOT}.incomplete
test "${incomplete}" != "${REMOTE_ROOT}"
ssh "${ssh_opts[@]}" "${host}" \
  "test ! -e '${REMOTE_ROOT}' && test ! -e '${incomplete}' && mkdir -p '${incomplete}/bundle'"
tar -C "${bundle_dir}" -cf - \
  README.md allocate.sh analyze_pair.py brill_global_48x32.coefficients \
  bundle.sha256 contract.env preflight_login.sh require_bound_contract.sh \
  run_allocation.sh self_test.sh source_input.athinput stage.sh \
  verify_predecessor.sh | \
  ssh "${ssh_opts[@]}" "${host}" "tar -C '${incomplete}/bundle' -xf -"
ssh "${ssh_opts[@]}" "${host}" \
  "bash '${incomplete}/bundle/preflight_login.sh' '${incomplete}' && mv '${incomplete}' '${REMOTE_ROOT}'"
printf 'STAGE_AND_LOGIN_PREFLIGHT_PASS %s\n' "${REMOTE_ROOT}"

#!/bin/bash
set -euo pipefail
local_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${local_root}/contract.env"
bash "${local_root}/require_bound_contract.sh"
remote=hzhu@perlmutter.nersc.gov
identity=/home/hzhu/.ssh/nersc
remote_incomplete=${REMOTE_ROOT}.incomplete
ssh_options=(-i "${identity}" -o IdentitiesOnly=yes -o BatchMode=yes)
files=(README.md allocate.sh bundle.sha256 contract.env finalize_campaign.sh
  preflight_login.sh require_bound_contract.sh run_allocation.sh stage.sh
  test_bundle.sh validate_accounting.awk)
(cd "${local_root}" && sha256sum -c bundle.sha256 >/dev/null)
test -z "$(find "${local_root}" \( -type d -name __pycache__ -o \
  -type f -name '*.pyc' \) -print -quit)"
remote_command="set -e; test ! -e '${REMOTE_ROOT}';"
remote_command+=" test ! -e '${remote_incomplete}';"
remote_command+=" mkdir -p '${remote_incomplete}/bundle';"
remote_command+=" tar -xf - -C '${remote_incomplete}/bundle';"
remote_command+=" cd '${remote_incomplete}/bundle'; sha256sum -c bundle.sha256;"
remote_command+=" bash preflight_login.sh; cd /;"
remote_command+=" mv -T --no-clobber '${remote_incomplete}' '${REMOTE_ROOT}';"
remote_command+=" cd '${REMOTE_ROOT}'; sha256sum -c preflight.sha256;"
remote_command+=" test -f '${REMOTE_ROOT}/PREFLIGHT_COMPLETE'"
(cd "${local_root}" && tar -cf - "${files[@]}") | \
  ssh "${ssh_options[@]}" "${remote}" "${remote_command}"
printf 'staged=%s\n' "${REMOTE_ROOT}"

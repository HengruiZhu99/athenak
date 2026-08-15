#!/bin/bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${root}/contract.env"
bash "${root}/require_bound_contract.sh"
(cd "${root}" && sha256sum -c bundle.sha256 >/dev/null)
for script in allocate.sh finalize_campaign.sh preflight_login.sh \
  require_bound_contract.sh run_allocation.sh stage.sh test_bundle.sh; do
  bash -n "${root}/${script}"
done
test -z "$(find "${root}" \( -type d -name __pycache__ -o \
  -type f -name '*.pyc' \) -print -quit)"
test "$(awk 'BEGIN{n=0} /run-case --output/{n++} END{print n}' \
  "${root}/run_allocation.sh")" -eq 1
grep -Fq 'for case_name in moving_puncture_h32 moving_puncture_h48 moving_puncture_h64' \
  "${root}/run_allocation.sh"
! rg -n 'zero_shift|telegraph_lapse = true|shift_Gamma = 0' "${root}" \
  --glob '!bundle.sha256' --glob '!test_bundle.sh'
grep -Fq -- "--constraint='gpu&hbm40g'" "${root}/allocate.sh"
grep -Fq -- '--cpus-per-task=8' "${root}/allocate.sh"
grep -Fq -- '--gpus-per-node=4' "${root}/allocate.sh"
grep -Fq 'EXPECTED_NUMBERED_STEPS=3' "${root}/contract.env"
grep -Fq 'PYTHON_MODULE=python/3.12-26.1.0' "${root}/contract.env"
grep -Fq 'EXPECTED_SYMPY_VERSION=1.14.0' "${root}/contract.env"
grep -Fq 'EXPECTED_H5PY_VERSION=3.15.1' "${root}/contract.env"
grep -Fq 'EXPECTED_AXIS_TEST_REPAIR_PATH=tst/unit/z4c/cartoon_axis_boundary_test.cpp' \
  "${root}/contract.env"
grep -Fq 'V4_JOB_ID=56880760' "${root}/contract.env"
printf 'half-plane Kerr bundle self-test passed\n'

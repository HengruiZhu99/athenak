#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"

test "$(sha256sum "${PREDECESSOR_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_DETACHED_SHA256}"
(cd "${PREDECESSOR_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null)
test "$(sha256sum "${PREDECESSOR_ROOT}/allocation/sacct-settled.psv" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_SACCT_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/run/comparison.json" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_COMPARISON_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_TAU1}/brill_fig3_fixed_eta2_tau1_kappa1_l20_n128.z4c.user.hst" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_TAU1_HISTORY_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_TAU2}/brill_fig3_fixed_eta2_tau2_kappa2_l20_n128.z4c.user.hst" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_TAU2_HISTORY_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_TAU1}/run.log" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_TAU1_LOG_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_TAU2}/run.log" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_TAU2_LOG_SHA256}"
grep -Fq "${EXPECTED_PREDECESSOR_JOB_ID}|cartoon-r4-fixedshift-l20-taupair-mpi4-v3|FAILED|1:0" \
  "${PREDECESSOR_ROOT}/allocation/sacct-settled.psv"
grep -Fq 'rejected 123902 parent stencils and 0 limited sibling groups' \
  "${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_TAU1}/run.log"
grep -Fq 'axis-central diagnostic support is nonfinite or invalid' \
  "${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_TAU2}/run.log"
python_bin=${PYTHON_BIN}
if [[ ! -x "${python_bin}" ]]; then
  python_bin=$(command -v python3)
fi
"${python_bin}" -B - "${PREDECESSOR_ROOT}/run/comparison.json" <<'PY'
import json, pathlib, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["both_cases_attempted"] is True
assert d["qualification_claim"] is False
assert [(c["name"], c["tau"], c["kappa"]) for c in d["cases"]] == [
    ("fixed_eta2_tau1_kappa1_l20", 1.0, 1.0),
    ("fixed_eta2_tau2_kappa2_l20", 2.0, 2.0),
]
assert [(c["exit_code"], c["max_refinement_level_reached"]) for c in d["cases"]] == [(1,20),(1,20)]
assert d["all_rank_bindings_verified"] is True
PY

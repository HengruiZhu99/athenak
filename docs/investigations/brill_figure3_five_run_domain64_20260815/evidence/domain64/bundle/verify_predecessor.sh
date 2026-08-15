#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"

# The predecessor outer wrapper regenerates the bulk manifest after the
# allocation has already released.  Authenticate the complete compact science
# record directly so launch is not coupled to rereading 131 GiB of raw slices.
test -f "${bundle_dir}/predecessor-selected.sha256"
(cd "${PREDECESSOR_ROOT}" && \
  sha256sum -c "${bundle_dir}/predecessor-selected.sha256" >/dev/null)
test "$(sha256sum "${PREDECESSOR_ROOT}/bundle/bundle.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_BUNDLE_MANIFEST_SHA256}"
test "$(sha256sum "${PREDECESSOR_ROOT}/preflight.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PREDECESSOR_PREFLIGHT_MANIFEST_SHA256}"

ko002=${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_KO002}
ko05=${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_KO05}
zero_shift=${PREDECESSOR_ROOT}/run/cases/${PREDECESSOR_CASE_ZERO_SHIFT}
files=(
  "${PREDECESSOR_ROOT}/allocation/sacct-settled.psv"
  "${PREDECESSOR_ROOT}/run/comparison.json"
  "${ko002}/brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko002_n128.z4c.user.hst"
  "${ko05}/brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko05_n128.z4c.user.hst"
  "${zero_shift}/brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_n128.z4c.user.hst"
  "${ko002}/run.log"
  "${ko05}/run.log"
  "${zero_shift}/run.log"
)
for path in "${files[@]}"; do
  test -f "${path}"
done

python_bin=${PYTHON_BIN}
if [[ ! -x "${python_bin}" ]]; then
  python_bin=$(command -v python3)
fi
"${python_bin}" -B - \
  "${PREDECESSOR_ROOT}/allocation/sacct-settled.psv" \
  "${PREDECESSOR_ROOT}/run/comparison.json" \
  "${EXPECTED_PREDECESSOR_JOB_ID}" "${EXPECTED_PREDECESSOR_JOB_NAME}" <<'PY'
import json, pathlib, sys
sacct_path, comparison_path, job_id, job_name = sys.argv[1:]
rows=[line.split("|") for line in pathlib.Path(sacct_path).read_text().splitlines() if line]
parent=[row for row in rows if row[0] == job_id]
assert len(parent) == 1
assert parent[0][1] == job_name
assert parent[0][2] in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}
steps=[row for row in rows if row[0].startswith(job_id + ".") and row[0].split(".")[-1].isdigit()]
assert [row[0] for row in steps] == [f"{job_id}.{i}" for i in range(3)]
assert all(row[2] in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"} for row in steps)
d=json.loads(pathlib.Path(comparison_path).read_text())
assert d["schema"] == "athenak_brill_l20_tau1_nocd_ko_shift_trio_mpi4_v1"
assert d["all_cases_attempted"] is True
assert d["qualification_claim"] is False
assert [(c["name"], c["ko_dissipation"], c["shift_condition"]) for c in d["cases"]] == [
    ("fixed_eta2_tau1_kappa1_l20_nocd_ko002", 0.02, "fixed_gamma_driver_eta2"),
    ("fixed_eta2_tau1_kappa1_l20_nocd_ko05", 0.5, "fixed_gamma_driver_eta2"),
    ("zero_shift_tau1_kappa1_l20_nocd_ko05", 0.5, "zero_shift"),
]
assert all(c["exit_code"] is not None for c in d["cases"])
assert all(c["constraint_damping"] is False for c in d["cases"])
assert d["all_rank_bindings_verified"] is True
PY

sha256sum "${bundle_dir}/predecessor-selected.sha256" "${files[@]}"

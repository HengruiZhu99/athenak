#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
failed_evidence=${campaign_root}/evidence/phase5-brill-initial
evidence_root=${campaign_root}/evidence/phase5-brill-initial-analysis-recovery1
run_root=${campaign_root}/runs/phase5-brill-initial
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
python_packages=${campaign_root}/python-packages
analyzer=${campaign_root}/analyze_brill_initial_data.py

test "$(cat "${failed_evidence}/exit-status.txt")" = 1
test ! -e "${evidence_root}"
for resolution in 128 256 512; do
  test -f "${run_root}/N${resolution}/rhs.rank000000.csv"
  test -f "${run_root}/N${resolution}/z4c_vc_brill_direct_fixed.constraints.dat"
done
mkdir -p "${evidence_root}"

finish() {
  status=$?
  set +e
  printf '%s\n' "${status}" > "${evidence_root}/exit-status.txt"
  find "${evidence_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 -r sha256sum > "${evidence_root}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

export PYTHONPATH=${python_packages}
"${python_bin}" "${analyzer}" \
  --run 128 "${run_root}/N128" \
  --run 256 "${run_root}/N256" \
  --run 512 "${run_root}/N512" \
  --output-dir "${evidence_root}/analysis" \
  > "${evidence_root}/analysis.log" 2>&1

"${python_bin}" - "${evidence_root}/analysis/summary.json" <<'PY'
import json, math, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["schema"] == "z4c_vc_brill_initial_data_v2"
assert len(summary["metrics"]) == 3
for metric in summary["metrics"]:
    assert metric["shared_state_max_spread"] == 0.0
    assert metric["min_chi"] > 0.0 and metric["min_alpha"] > 0.0
    assert min(metric["minimum_spd_pivots"]) > 0.0
    assert all(math.isfinite(value) for item in metric["constraints"].values()
               for value in item.values())
for comparison in summary["common_node_field_comparisons"]:
    assert comparison["direct_initialized_field_linf"] < 1.0e-12
PY

find "${run_root}" -name 'rhs.rank*.csv' -print0 | sort -z | \
  xargs -0 -r sha256sum > "${evidence_root}/raw-rhs-before-compression.sha256"
find "${run_root}" -name 'rhs.rank*.csv' -print0 | sort -z | xargs -0 -r gzip -9
find "${run_root}" -type f -print0 | sort -z | xargs -0 -r sha256sum \
  > "${evidence_root}/run-products.sha256"
sha256sum "${failed_evidence}/SHA256SUMS" \
  > "${evidence_root}/failed-attempt-manifest.sha256"

printf '%s\n' \
  'DIRECT_VC_BRILL_INITIAL_DATA_BOUNDED_AUDIT_PASSED_FROM_EXISTING_RUNS' \
  > "${evidence_root}/verdict.txt"

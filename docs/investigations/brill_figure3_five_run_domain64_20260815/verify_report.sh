#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${root}"

test -f evidence/domain16/predecessor-selected.sha256
(cd evidence/domain16 && sha256sum -c predecessor-selected.sha256 >/dev/null)

test -f evidence/domain64/domain64-selected.sha256
(cd evidence/domain64 && sha256sum -c domain64-selected.sha256 >/dev/null)

python3 -B - <<'PY'
import json
import math
from pathlib import Path

root = Path.cwd()
summary = json.loads((root / "analysis/five_case_summary.json").read_text())
assert summary["schema"] == "athenak_brill_figure3_five_run_domain_comparison_v1"
assert summary["qualification_claim"] is False
assert summary["figure3_reproduction_claim"] is False
assert set(summary["cases"]) == {
    "d16_fixed_ko002",
    "d16_fixed_ko05",
    "d16_zero_ko05",
    "d64_fixed_ko05",
    "d64_zero_ko05",
}
for case in summary["cases"].values():
    result = case["result"]
    assert result["qualification_claim"] is False
    assert isinstance(result["exit_code"], int)
    assert case["max_refinement_level"] >= 0
    assert case["max_meshblocks"] > 0
    for value in case["last_finite_history"].values():
        assert math.isfinite(float(value))

for stem in (
    "figure3_five_case_overlay",
    "constraints_five_case",
    "gauge_amr_five_case",
    "boundary_distance_pairwise",
):
    assert (root / "figures" / f"{stem}.png").is_file()
    assert (root / "figures" / f"{stem}.pdf").is_file()

assert (root / "report.pdf").is_file()
assert (root / "analysis/generated_results.tex").is_file()
assert not any(root.rglob("__pycache__"))
assert not any(root.rglob("*.pyc"))
PY

test -f SHA256SUMS
test -f SHA256SUMS.sha256
sha256sum -c SHA256SUMS >/dev/null
sha256sum -c SHA256SUMS.sha256 >/dev/null

printf 'FIVE_CASE_REPORT_VERIFY_PASS\n'

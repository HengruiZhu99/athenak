#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${root}"

test -f SHA256SUMS
test -f SHA256SUMS.sha256
sha256sum -c SHA256SUMS >/dev/null
sha256sum -c SHA256SUMS.sha256 >/dev/null

test -z "$(find . \( -type d -name __pycache__ -o -type f -name '*.pyc' \) -print)"
test "$(find figures -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 4
test "$(find figures -maxdepth 1 -type f -name '*.pdf' | wc -l)" -eq 4
find figures -maxdepth 1 -type f -size +0c | grep -q .

jq -e '
  .qualification_claim == false and
  .disposition == "restriction_inconsistency_fixed_but_late_instability_not_cured" and
  .source.commit == "345dd31d59cebd9c0c7231be43dcc6a72524bcc7" and
  .verification.full_remote_root_sha256sum_check == "pass" and
  .interpretation.restriction_mismatch_contributed_to_typical_amr_jumps == true and
  .interpretation.late_instability_eliminated == false
' data/report_summary.json >/dev/null

jq -e '
  .qualification_claim == false and
  .constraint_columns_are_squared_volume_integrals == true and
  .plotted_constraint_quantity == "sqrt(history column)" and
  .cases.post_n128.terminal.time == 16.728488016128324 and
  .cases.post_n256.terminal.time == 11.955865198373266
' data/comparison_summary.json >/dev/null

jq -e '
  .selected_verification_pass == true and
  .selected_files_verified == 40 and
  .root_manifest_sha256 == "e846e03bf94cb10085f8d161368f86958dbc66eeea4f5a4bf30cfc11e82ddfbc"
' data/terminal_evidence.json >/dev/null

printf '%s\n' REPORT_VERIFICATION_PASS

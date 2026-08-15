#!/bin/bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  printf 'usage: finalize_campaign.sh ROOT SOURCE HEAD TREE\n' >&2
  exit 2
fi
campaign_root=$1
source_root=$2
expected_head=$3
expected_tree=$4
test -d "${campaign_root}/allocation"
test ! -e "${campaign_root}/SHA256SUMS"
test ! -e "${campaign_root}/SHA256SUMS.sha256"
status_path=${campaign_root}/allocation/source-status-final.txt
observed_head=$(git -C "${source_root}" rev-parse HEAD)
observed_tree=$(git -C "${source_root}" rev-parse HEAD^{tree})
porcelain=$(git -C "${source_root}" status --short)
clean=false
if [[ ${observed_head} == "${expected_head}" && \
      ${observed_tree} == "${expected_tree}" && -z ${porcelain} ]]; then
  clean=true
fi
{
  printf 'path=%s\nexpected_head=%s\nobserved_head=%s\n' \
    "${source_root}" "${expected_head}" "${observed_head}"
  printf 'expected_tree=%s\nobserved_tree=%s\nclean=%s\n' \
    "${expected_tree}" "${observed_tree}" "${clean}"
  printf 'status_begin\n%s\nstatus_end\n' "${porcelain}"
  git -C "${source_root}" submodule status
} > "${status_path}"
(
  cd "${campaign_root}"
  find . -type f ! -path './SHA256SUMS' ! -path './SHA256SUMS.sha256' \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.sha256
  sha256sum -c SHA256SUMS >/dev/null
  sha256sum -c SHA256SUMS.sha256 >/dev/null
)
test "${clean}" = true

#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${here}"

tmp=$(mktemp -d)
trap 'rm -rf -- "${tmp}"' EXIT

export SOURCE_DATE_EPOCH=1786665600
export FORCE_SOURCE_DATE=1

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory "${tmp}" figure3_shift_damping_report.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory "${tmp}" figure3_shift_damping_report.tex >/dev/null

if LC_ALL=C grep -Eq 'Warning|Overfull|Underfull|undefined' \
    "${tmp}/figure3_shift_damping_report.log"; then
  grep -En 'Warning|Overfull|Underfull|undefined' \
    "${tmp}/figure3_shift_damping_report.log" >&2
  exit 1
fi

install -m 0644 "${tmp}/figure3_shift_damping_report.pdf" \
  figure3_shift_damping_report.pdf
sha256sum figure3_shift_damping_report.pdf

#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${root}"
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1786795200}
python3 -B plot_five_cases.py
python3 -B render_results_tex.py
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/pass1.stdout.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/pass2.stdout.log
cp build/report.pdf report.pdf
grep -E 'LaTeX Warning|Overfull|Undefined control sequence|multiply defined' build/report.log > build/warning_audit.txt || true
test ! -s build/warning_audit.txt
printf 'FIVE_CASE_REPORT_BUILD_PASS\n'

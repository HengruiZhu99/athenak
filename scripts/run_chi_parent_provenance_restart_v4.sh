#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-parent-provenance-ac75c8d3-v1-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
evidence=${root}/evidence
failed_run=${root}/run/n256-replay-provenance-v3
run=${root}/run/n256-replay-provenance-v4
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
restart=${root}/input/n256-replay-cycle4096.rst
coeff=${root}/input/brill_global_48x32.coefficients

test "${SLURM_JOB_ID:-}" = 57214220
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "$(git -C "${source_root}" rev-parse HEAD)" = \
  ac75c8d348da91b38cbc6855b5fba51cd3089663
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = \
  6284882bd06e8db379495675aba7a4f153fb4afa
test "$(sha256sum "${restart}" | awk '{print $1}')" = \
  2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9
test "$(cat "${failed_run}/exit-status.txt")" = 1
grep -F 'fresh output directory already exists' "${failed_run}/run.log" >/dev/null
grep -Fx 'CUDA_LAYOUT_FIX_BUILD_AND_TESTS_PASS' \
  "${evidence}/layout-fix/status.txt" >/dev/null
test -s "${evidence}/v3-output-cwd-failure/binding.sha256"
test ! -e "${run}"
mkdir -p "${run}/bindings" "${run}/parse-smoke"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final-v4"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${evidence}/kokkos-status.final-v4"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
    -print0 | sort -z | xargs -0 sha256sum > "${root}/SHA256SUMS"
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU

parse_command=("${build_root}/src/athena" -r "${restart}" -n
  -d "${run}" job/basename=N256-replay-provenance-v4
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=N256-replay-provenance-v4-constraints.dat
  output2/dcycle=99999999 output3/dcycle=99999999
  output4/dcycle=99999999 output5/dcycle=99999999
  output6/dcycle=99999999 output7/dcycle=99999999
  output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${parse_command[@]}" > "${run}/parse-smoke/command.txt"
printf '\n' >> "${run}/parse-smoke/command.txt"
(cd "${run}" && "${parse_command[@]}") > \
  "${run}/parse-smoke/parameter-dump.txt" 2> "${run}/parse-smoke/stderr.txt"
grep -F '<amr_history_restart>' "${run}/parse-smoke/parameter-dump.txt" >/dev/null
grep -F 'mode               = replay' "${run}/parse-smoke/parameter-dump.txt" >/dev/null
grep -F 'last_applied_event = 9' "${run}/parse-smoke/parameter-dump.txt" >/dev/null
grep -F 'next_event         = 10' "${run}/parse-smoke/parameter-dump.txt" >/dev/null
printf 'RESTART_ONLY_PARSE_PASS\n' > "${run}/parse-smoke/status.txt"

export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=9.3515625
export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_parent_provenance
command=("${python_bin}" "${wrapper}" --evidence-dir "${run}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${restart}"
  -d "${run}" job/basename=N256-replay-provenance-v4
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=N256-replay-provenance-v4-constraints.dat
  output2/dcycle=99999999 output3/dcycle=99999999
  output4/dcycle=99999999 output5/dcycle=99999999
  output6/dcycle=99999999 output7/dcycle=99999999
  output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${command[@]}" > "${run}/command.txt"
printf '\n' >> "${run}/command.txt"
status=0
(cd "${run}" && "${command[@]}") > "${run}/run.log" 2>&1 || status=$?
printf '%s\n' "${status}" > "${run}/exit-status.txt"
sha256sum "${run}/command.txt" "${run}/run.log" \
  "${run}/exit-status.txt" > "${run}/terminal.sha256"
test "${status}" = 1
test -s "${run}/chi_parent_provenance/phase1_disposition.json"
test -s "${run}/chi_parent_provenance/first_invalid_coarse_cell.json"
test -s "${run}/chi_parent_provenance/unique_invalid_coarse_cells.csv"
"${python_bin}" - "${run}/chi_parent_provenance" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in root.glob("*.json"):
    json.loads(path.read_text())
rows = list(csv.DictReader((root / "unique_invalid_coarse_cells.csv").open()))
if not rows:
    raise SystemExit("no unique invalid coarse cells were recorded")
summary = json.loads((root / "first_invalid_coarse_cell.json").read_text())
if summary["unique_invalid_coarse_cells"] != len(rows):
    raise SystemExit("unique invalid cell count mismatch")
print("CHI_PARENT_PROVENANCE_STRICT_OUTPUT_PASS")
PY
printf 'PHASE1_DIAGNOSTIC_COMPLETE\n' > "${run}/diagnostic-status.txt"

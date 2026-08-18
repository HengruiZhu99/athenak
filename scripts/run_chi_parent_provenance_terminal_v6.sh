#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-parent-provenance-ac75c8d3-v1-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
evidence=${root}/evidence
run=${root}/run/n256-replay-provenance-terminal-v6
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
grep -F '"classification":"ACTIVE_FINE_CHI_FAILURE"' \
  "${root}/run/n256-replay-provenance-v5/chi_parent_provenance/phase1_disposition.json" \
  >/dev/null
test -s "${evidence}/v5-active-fine-disposition/binding.sha256"
test ! -e "${run}"
mkdir -p "${run}/bindings" "${evidence}/active-cell-dump"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final-v6"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${evidence}/kokkos-status.final-v6"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
    -print0 | sort -z | xargs -0 sha256sum > "${root}/SHA256SUMS"
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
sha256sum "${source_root}/src/z4c/chi_parent_provenance.cpp" > \
  "${evidence}/active-cell-dump/source.sha256"
cmake --build "${build_root}" --parallel 8 --target athena > \
  "${evidence}/active-cell-dump/build.log" 2> \
  "${evidence}/active-cell-dump/build.err"
ctest --test-dir "${build_root}" --output-on-failure \
  -R '^athena\.(amr_history_format|z4c_chi_prolongation|z4c_coarse_cache_ownership|z4c_cartoon_amr_static|z4c_amr_derefine_factor_static)$' \
  > "${evidence}/active-cell-dump/focused-tests.log" 2>&1
grep -F '100% tests passed, 0 tests failed out of 5' \
  "${evidence}/active-cell-dump/focused-tests.log" >/dev/null
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" > \
  "${evidence}/active-cell-dump/build-products.sha256"
printf 'CUDA_ACTIVE_CELL_DUMP_BUILD_AND_TESTS_PASS\n' > \
  "${evidence}/active-cell-dump/status.txt"

export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=10.5
export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_parent_provenance
command=("${python_bin}" "${wrapper}" --evidence-dir "${run}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${restart}"
  -d "${run}" job/basename=N256-replay-provenance-terminal-v6
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=N256-replay-provenance-terminal-v6-constraints.dat
  output2/dcycle=99999999 output3/dcycle=99999999
  output4/dcycle=99999999 output5/dcycle=99999999
  output6/dcycle=99999999 output7/dcycle=99999999
  output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${command[@]}" > "${run}/command.txt"
printf '\n' >> "${run}/command.txt"
status=0
(cd "${run}" && "${command[@]}") > "${run}/run.log" 2>&1 || status=$?
printf '%s\n' "${status}" > "${run}/exit-status.txt"
test "${status}" = 1
test -s "${run}/chi_parent_provenance/phase1_disposition.json"
test -s "${run}/chi_parent_provenance/active_fine_failure.csv"
"${python_bin}" - "${run}/chi_parent_provenance" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
disposition = json.loads((root / "phase1_disposition.json").read_text())
assert disposition == {
    "classification": "ACTIVE_FINE_CHI_FAILURE",
    "cycle": 5546,
    "stage": 3,
    "time_hex": "0x1.5124ccccccd9bp+3",
    "active_nonpositive": 2,
    "active_nonfinite": 0,
}
rows = list(csv.DictReader((root / "active_fine_failure.csv").open()))
assert len(rows) == 2
assert all(row["classification"] == "negative" for row in rows)
assert all(int(row["cycle"]) == 5546 and int(row["rk_stage"]) == 3 for row in rows)
print("ACTIVE_FINE_CELL_DUMP_STRICT_PASS")
PY
sha256sum "${run}/command.txt" "${run}/run.log" \
  "${run}/exit-status.txt" \
  "${run}/chi_parent_provenance/phase1_disposition.json" \
  "${run}/chi_parent_provenance/active_fine_failure.csv" > \
  "${run}/terminal.sha256"
printf 'PHASE1_ACTIVE_FINE_FAILURE_COMPLETE\n' > "${run}/diagnostic-status.txt"

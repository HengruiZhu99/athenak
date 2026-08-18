#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-parent-provenance-ac75c8d3-v1-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
evidence=${root}/evidence
run=${root}/run/n256-replay-provenance
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
iris_root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/source/iris
iris_archive=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/build/iris-host/src/libiris_athenak_interpolator.a
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
restart=${root}/input/n256-replay-cycle4096.rst
history=${root}/input/n128-authority-hierarchy.jsonl
coeff=${root}/input/brill_global_48x32.coefficients
input=${root}/input/pilot.athinput

test "${SLURM_JOB_ID:-}" = 57214220
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "$(git -C "${source_root}" rev-parse HEAD)" = \
  ac75c8d348da91b38cbc6855b5fba51cd3089663
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = \
  6284882bd06e8db379495675aba7a4f153fb4afa
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = \
  6739bc623081648af9e752b616d9671527922cbf
test "$(sha256sum "${restart}" | awk '{print $1}')" = \
  2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9
test "$(sha256sum "${history}" | awk '{print $1}')" = \
  d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555
test "$(sha256sum "${coeff}" | awk '{print $1}')" = \
  ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
test "$(sha256sum "${input}" | awk '{print $1}')" = \
  edced480bbd934347aa80152dda4c164c4b6fd59c2a7abe764ac990983004791
test ! -e "${build_root}"
test ! -e "${run}"

mkdir -p "${evidence}" "${run}/bindings"
finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${evidence}/kokkos-status.final"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
    -print0 | sort -z | xargs -0 sha256sum > "${root}/SHA256SUMS"
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export COLLAPSE_ROOT=${root}
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${evidence}/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${evidence}/node.txt"
nvidia-smi -L > "${evidence}/nvidia-smi-L.txt"

git -C "${source_root}" diff --binary HEAD > "${evidence}/diagnostic.patch"
sha256sum "${evidence}/diagnostic.patch" > "${evidence}/diagnostic.patch.sha256"
git -C "${source_root}" status --short > "${evidence}/source-status.initial"

configure=(cmake -S "${source_root}" -B "${build_root}"
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC
  -DPROBLEM=built_in_pgens -DBUILD_TESTING=ON -DAthena_BUILD_UNIT_TESTS=ON
  -DAthena_ENABLE_IRISK_INTERPOLATOR=ON -DIRISK_ROOT="${iris_root}"
  -DIRISK_INTERPOLATOR_LIBRARY="${iris_archive}"
  -DAthena_SINGLE_PRECISION=OFF -DAthena_ENABLE_MPI=ON
  -DAthena_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DKokkos_ENABLE_OPENMP=ON
  -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ARCH_AMPERE80=ON)
printf '%q ' "${configure[@]}" > "${evidence}/configure-command.txt"
printf '\n' >> "${evidence}/configure-command.txt"
/usr/bin/time -p -o "${evidence}/configure-time.txt" "${configure[@]}" \
  > "${evidence}/configure.log" 2> "${evidence}/configure.err"

build=(cmake --build "${build_root}" --parallel 8 --target athena
  athena_amr_history_format_unit_test athena_z4c_chi_prolongation_unit_test
  athena_z4c_coarse_cache_ownership_unit_test)
printf '%q ' "${build[@]}" > "${evidence}/build-command.txt"
printf '\n' >> "${evidence}/build-command.txt"
/usr/bin/time -p -o "${evidence}/build-time.txt" "${build[@]}" \
  > "${evidence}/build.log" 2> "${evidence}/build.err"

ctest --test-dir "${build_root}" --output-on-failure \
  -R '^athena\.(amr_history_format|z4c_chi_prolongation|z4c_coarse_cache_ownership|z4c_cartoon_amr_static|z4c_amr_derefine_factor_static)$' \
  > "${evidence}/focused-tests.log" 2>&1
grep -F '100% tests passed, 0 tests failed out of 5' \
  "${evidence}/focused-tests.log" >/dev/null
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" \
  > "${evidence}/build-products.sha256"
printf 'BUILD_AND_FOCUSED_TESTS_PASS\n' > "${evidence}/build-status.txt"

export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=9.3515625
export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_parent_provenance
command=("${python_bin}" "${wrapper}" --evidence-dir "${run}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${restart}" -i "${input}"
  -d "${run}" job/basename=N256-replay-provenance
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=N256-replay-provenance-constraints.dat
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file="${history}"
  output2/dcycle=99999999 output3/dcycle=99999999
  output4/dcycle=99999999 output5/dcycle=99999999
  output6/dcycle=99999999 output7/dcycle=99999999
  output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${command[@]}" > "${run}/command.txt"
printf '\n' >> "${run}/command.txt"
status=0
"${command[@]}" > "${run}/run.log" 2>&1 || status=$?
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

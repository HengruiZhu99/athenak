#!/bin/bash
set -euo pipefail

old_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
campaign_root=/pscratch/sd/h/hzhu/z4c-vc-cartoon-axis-boundary-20260823
source_root=${old_root}/source/athenak
build_root=${old_root}/build/current-cuda-mpi
evidence_root=${campaign_root}/evidence/fixed-grid-gate
run_root=${campaign_root}/runs/fixed-grid-gate
required_source_fix=1392f5c472353fec1cdc44108b403a316f33fc46
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
coefficient=${old_root}/authority/brill_global_48x32.coefficients
input=${campaign_root}/fixed_grid_brill_dense.athinput
athena=${build_root}/src/athena
manufactured=${build_root}/athena_cartoon_derivatives_unit_test
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
git -C "${source_root}" merge-base --is-ancestor "${required_source_fix}" HEAD
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${athena}"
test -x "${manufactured}"
test -f "${coefficient}"
test -f "${input}"
test ! -e "${evidence_root}"
test ! -e "${run_root}"
mkdir -p "${evidence_root}" "${run_root}"

finish() {
  status=$?
  set +e
  printf '%s\n' "${status}" > "${evidence_root}/exit-status.txt"
  find "${evidence_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 -r sha256sum > "${evidence_root}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
scontrol show job "${SLURM_JOB_ID}" > "${evidence_root}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
env | sort > "${evidence_root}/environment.txt"
nvidia-smi -L > "${evidence_root}/nvidia-smi.txt"
git -C "${source_root}" status --short > "${evidence_root}/source-status.txt"
git -C "${source_root}" rev-parse HEAD > "${evidence_root}/source-commit.txt"
sha256sum "${athena}" "${manufactured}" "${build_root}/CMakeCache.txt" \
  "${coefficient}" "${input}" > "${evidence_root}/authority-products.sha256"
grep -E 'Athena_ENABLE_(MPI|OPENMP|IRISK)|IRISK_ROOT|IRISK_INTERPOLATOR_LIBRARY|Kokkos_ENABLE_(CUDA|OPENMP|SERIAL)|Kokkos_ARCH_AMPERE80|CMAKE_BUILD_TYPE' \
  "${build_root}/CMakeCache.txt" > "${evidence_root}/cache-contract.txt"

"${manufactured}" > "${evidence_root}/manufactured-cartoon.log" 2>&1
grep -F 'Cartoon derivative manufactured-oracle tests passed' \
  "${evidence_root}/manufactured-cartoon.log" >/dev/null

ctest_regex='^athena[.](z4c_vc_cartoon_derivatives_o2|z4c_vc_cartoon_derivatives_o4|z4c_vc_cartoon_derivatives_o6|z4c_vc_cartoon_axis_scalar|z4c_vc_cartoon_axis_vector|z4c_vc_cartoon_axis_tensor|z4c_vc_cartoon_axis_adm|z4c_vc_cartoon_axis_constraint|z4c_vc_cartoon_axis_rhs_regularity|z4c_cartoon_production_kernels_cuda_required|z4c_vc_restart_2d_cartoon|z4c_vc_output_2d_cartoon)$'
ctest --test-dir "${build_root}" --output-on-failure -j 1 -R "${ctest_regex}" \
  > "${evidence_root}/ctest-cuda-selected.log" 2>&1
grep -F '100% tests passed' "${evidence_root}/ctest-cuda-selected.log" >/dev/null

for resolution in 128 256 512; do
  meshblock=$((resolution / 4))
  run=${run_root}/N${resolution}
  basename=z4c_vc_axis_fixed_N${resolution}
  mkdir -p "${run}"
  start=$(date +%s)
  "${athena}" -i "${input}" -d "${run}" \
    job/basename="${basename}" \
    mesh/nx1="${resolution}" mesh/nx2="$((2 * resolution))" \
    meshblock/nx1="${meshblock}" meshblock/nx2="${meshblock}" \
    problem/brill_global_coefficients_file="${coefficient}" \
    problem/constraint_summary_file="${run}/${basename}.constraints.dat" \
    > "${evidence_root}/N${resolution}.stdout.log" \
    2> "${evidence_root}/N${resolution}.stderr.log"
  end=$(date +%s)
  printf '%s\n' "$((end - start))" \
    > "${evidence_root}/N${resolution}.wall-seconds.txt"
  test ! -e "${run}/z4c_state_failure.json"
  history=${run}/${basename}.z4c.user.hst
  test -f "${history}"
  "${python_bin}" - "${history}" "${resolution}" <<'PY'
import math, sys
path, resolution = sys.argv[1], int(sys.argv[2])
lines = open(path, encoding="utf-8").read().splitlines()
header = next(line for line in lines if line.startswith("#  [1]="))
labels = [part.split("=")[-1].strip() for part in header[2:].split("[") if "]=" in part]
rows = [[float(value) for value in line.split()]
        for line in lines if line and not line.startswith("#")]
assert rows and all(math.isfinite(value) for row in rows for value in row)
last = rows[-1]
assert last[labels.index("time")] >= 5.0 - 1.0e-12
assert last[labels.index("axisTau")] >= 3.0
print(resolution, last[labels.index("time")], last[labels.index("axisTau")])
PY
  find "${run}" -type f -print0 | sort -z | xargs -0 -r sha256sum \
    > "${evidence_root}/N${resolution}.products.sha256"
done

printf '%s\n' 'FIXED_GRID_BRILL_DENSE_N128_N256_N512_REACHED_AXIS_TAU_3' \
  > "${evidence_root}/verdict.txt"

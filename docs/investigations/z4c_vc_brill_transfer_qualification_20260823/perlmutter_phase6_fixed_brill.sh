#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
source_root=${campaign_root}/source/athenak
build_root=${campaign_root}/build/current-cuda-mpi
evidence_root=${campaign_root}/evidence/phase6-fixed-brill-retry1
run_root=${campaign_root}/runs/phase6-fixed-brill-retry1
expected_source=278b63a740a947de55ad8bdd1c333095c68fedcd
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
coefficient=${campaign_root}/authority/brill_global_48x32.coefficients
input=${campaign_root}/z4c_vc_brill_direct_fixed_phase6.athinput
athena=${build_root}/src/athena
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${athena}"
test -f "${coefficient}"
test -f "${input}"
grep -q '^Athena_ENABLE_IRISK_INTERPOLATOR:BOOL=ON$' \
  "${build_root}/CMakeCache.txt"
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
grep -E 'Athena_ENABLE_(MPI|OPENMP|IRISK)|IRISK_ROOT|IRISK_INTERPOLATOR_LIBRARY|Kokkos_ENABLE_(CUDA|OPENMP|SERIAL)|Kokkos_ARCH_AMPERE80|CMAKE_BUILD_TYPE' \
  "${build_root}/CMakeCache.txt" > "${evidence_root}/cache-contract.txt"
sha256sum "${athena}" "${build_root}/CMakeCache.txt" "${coefficient}" \
  "${input}" > "${evidence_root}/authority-products.sha256"

for resolution in 128 256 512; do
  meshblock=$((resolution / 4))
  run=${run_root}/N${resolution}
  basename=z4c_vc_brill_fixed_N${resolution}
  mkdir -p "${run}"
  start=$(date +%s)
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 \
    "${athena}" -i "${input}" -d "${run}" \
    job/basename="${basename}" \
    mesh/nx1="${resolution}" mesh/nx2="$((2 * resolution))" \
    meshblock/nx1="${meshblock}" meshblock/nx2="${meshblock}" \
    problem/brill_global_coefficients_file="${coefficient}" \
    problem/constraint_summary_file="${run}/${basename}.constraints.dat" \
    > "${evidence_root}/N${resolution}.stdout.log" \
    2> "${evidence_root}/N${resolution}.stderr.log"
  end=$(date +%s)
  printf '%s\n' "$((end - start))" > "${evidence_root}/N${resolution}.wall-seconds.txt"
  test ! -e "${run}/z4c_state_failure.json"
  history=${run}/${basename}.z4c.user.hst
  test -f "${history}"
  state=$(find "${run}/bin" -name "${basename}.state.*.bin" -print | sort | tail -n 1)
  constraints=$(find "${run}/bin" -name "${basename}.constraints.*.bin" -print | sort | tail -n 1)
  restart=$(find "${run}/rst" -name "${basename}.*.rst" -print | sort | tail -n 1)
  test -n "${state}" && test -n "${constraints}" && test -n "${restart}"
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
assert last[labels.index("axisTau")] >= 3.0, (resolution, last[labels.index("axisTau")])
print(resolution, last[labels.index("time")], last[labels.index("axisTau")])
PY
  sha256sum "${history}" "${state}" "${constraints}" "${restart}" \
    > "${evidence_root}/N${resolution}.terminal-products.sha256"
done

find "${run_root}" -type f -print0 | sort -z | xargs -0 -r sha256sum \
  > "${evidence_root}/run-products.sha256"
printf '%s\n' 'FIXED_GRID_BRILL_N128_N256_N512_REACHED_AXIS_TAU_3' \
  > "${evidence_root}/verdict.txt"

#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
source_root=${campaign_root}/source/athenak
build_root=${campaign_root}/build/current-cuda-mpi
evidence_root=${campaign_root}/evidence/phase5-brill-initial
run_root=${campaign_root}/runs/phase5-brill-initial
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
python_packages=${campaign_root}/python-packages
expected_source=278b63a740a947de55ad8bdd1c333095c68fedcd
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
coefficient=${campaign_root}/authority/brill_global_48x32.coefficients
analyzer=${campaign_root}/analyze_brill_initial_data.py

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${build_root}/src/athena"
test -f "${coefficient}"
test -f "${analyzer}"
test "$(sha256sum "${coefficient}" | awk '{print $1}')" = \
  ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
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

export PYTHONPATH=${python_packages}
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
scontrol show job "${SLURM_JOB_ID}" > "${evidence_root}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
env | sort > "${evidence_root}/environment.txt"
nvidia-smi -L > "${evidence_root}/nvidia-smi.txt"
grep -E 'Athena_ENABLE_(MPI|OPENMP|IRISK)|IRISK_ROOT|IRISK_INTERPOLATOR_LIBRARY|Kokkos_ENABLE_(CUDA|OPENMP|SERIAL)|Kokkos_ARCH_AMPERE80|CMAKE_BUILD_TYPE' \
  "${build_root}/CMakeCache.txt" > "${evidence_root}/cache-contract.txt"
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" \
  "${coefficient}" \
  "${campaign_root}/source/irisk-authority/src/initial_data/handoff/irisk_athenak_spectral_interpolator.h" \
  "${campaign_root}/source/irisk-authority/build/serial-gcc/src/libiris_athenak_interpolator.a" \
  > "${evidence_root}/authority-products.sha256"

input=${source_root}/tst/inputs/z4c_vc_brill_direct_fixed.athinput
athena=${build_root}/src/athena
analysis_args=()
for resolution in 128 256 512; do
  meshblock=$((resolution / 4))
  stride=$((resolution / 128))
  run=${run_root}/N${resolution}
  mkdir -p "${run}"
  export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC=${run}/rhs
  export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC_STRIDE=${stride}
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 \
    "${athena}" -i "${input}" -d "${run}" \
    mesh/nx1="${resolution}" mesh/nx2="$((2 * resolution))" \
    meshblock/nx1="${meshblock}" meshblock/nx2="${meshblock}" \
    time/nlim=1 time/tlim=1.0 \
    problem/brill_global_coefficients_file="${coefficient}" \
    problem/constraint_summary_file=${run}/z4c_vc_brill_direct_fixed.constraints.dat \
    > "${evidence_root}/N${resolution}.stdout.log" \
    2> "${evidence_root}/N${resolution}.stderr.log"
  grep -F 'mode=direct_global_coefficients' \
    "${evidence_root}/N${resolution}.stdout.log" >/dev/null
  grep -F 'lapse=psi^-2' "${evidence_root}/N${resolution}.stdout.log" >/dev/null
  test -f "${run}/rhs.rank000000.csv"
  test -f "${run}/z4c_vc_brill_direct_fixed.constraints.dat"
  analysis_args+=(--run "${resolution}" "${run}")
done

"${python_bin}" "${analyzer}" "${analysis_args[@]}" \
  --output-dir "${evidence_root}/analysis" \
  > "${evidence_root}/analysis.log" 2>&1

"${python_bin}" - "${evidence_root}/analysis/summary.json" <<'PY'
import json, math, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(summary["metrics"]) == 3
for metric in summary["metrics"]:
    assert metric["shared_state_max_spread"] == 0.0
    assert metric["min_chi"] > 0.0 and metric["min_alpha"] > 0.0
    assert min(metric["minimum_spd_pivots"]) > 0.0
    assert all(math.isfinite(value) for item in metric["constraints"].values()
               for value in item.values())
for comparison in summary["common_node_field_comparisons"]:
    assert comparison["direct_initialized_field_linf"] < 1.0e-12
PY

find "${run_root}" -name 'rhs.rank*.csv' -print0 | sort -z | \
  xargs -0 -r sha256sum > "${evidence_root}/raw-rhs-before-compression.sha256"
find "${run_root}" -name 'rhs.rank*.csv' -print0 | sort -z | xargs -0 -r gzip -9
find "${run_root}" -type f -print0 | sort -z | xargs -0 -r sha256sum \
  > "${evidence_root}/run-products.sha256"

printf 'DIRECT_VC_BRILL_INITIAL_DATA_BOUNDED_AUDIT_PASSED\n' \
  > "${evidence_root}/verdict.txt"

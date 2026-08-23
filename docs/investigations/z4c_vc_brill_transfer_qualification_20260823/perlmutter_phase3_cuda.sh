#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
source_root=${campaign_root}/source/athenak
build_root=${campaign_root}/build/current-cuda-mpi
evidence_root=${campaign_root}/evidence/phase3-current-cuda
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
python_packages=${campaign_root}/python-packages
expected_source=11b9639336d7519121a4ddc4b5a7e1b4106897ec
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${build_root}/src/athena"
test ! -e "${evidence_root}"
mkdir -p "${evidence_root}"

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
grep -E 'Athena_ENABLE_(MPI|OPENMP)|Kokkos_ENABLE_(CUDA|OPENMP|SERIAL)|Kokkos_ARCH_AMPERE80|CMAKE_BUILD_TYPE' \
  "${build_root}/CMakeCache.txt" > "${evidence_root}/cache-contract.txt"
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" \
  > "${evidence_root}/build-products.sha256"

ctest_regex='^athena[.](z4c_cartoon_production_kernels_cuda_required|pdf_scatter_cuda_required|z4c_vc_dynamic_amr_2d_cartesian_o4|z4c_vc_dynamic_amr_3d_cartesian_o4|z4c_vc_restart_2d_cartesian|z4c_vc_restart_3d_cartesian|z4c_vc_output|z4c_vc_output_3d_cartesian|z4c_cc_selector_equivalence)$'
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=single:1 \
  --cpu-bind=cores --exact --kill-on-bad-exit=1 \
  ctest --test-dir "${build_root}" --output-on-failure -j 1 -R "${ctest_regex}" \
  > "${evidence_root}/ctest-selected.log" 2>&1
grep -F '100% tests passed' "${evidence_root}/ctest-selected.log" >/dev/null

convergence=${source_root}/tst/unit/z4c/z4c_vertex_dynamic_linear_wave_test.py
input2=${source_root}/tst/inputs/z4c_vc_dynamic_linear_wave_2d_cartesian.athinput
input3=${source_root}/tst/inputs/z4c_vc_dynamic_linear_wave_3d_cartesian.athinput
athena=${build_root}/src/athena
for dimensions in 2 3; do
  input=${input2}
  test "${dimensions}" -eq 3 && input=${input3}
  for transfer in 4 6; do
    label=o4-q${transfer}-${dimensions}d
    minimum=3
    test "${transfer}" -eq 4 && minimum=0
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
      --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 \
      "${python_bin}" "${convergence}" --athena "${athena}" --input "${input}" \
      --work-dir "${campaign_root}/runs/phase3-cuda-${label}" \
      --dimensions "${dimensions}" --order 4 --integrator rk4 \
      --vertex-prolongation-order "${transfer}" \
      --minimum-observed-order "${minimum}" --resolutions 16 32 64 \
      --json-output "${evidence_root}/${label}.json" \
      > "${evidence_root}/${label}.log" 2>&1
    grep -F 'PASS: native-VC' "${evidence_root}/${label}.log" >/dev/null
  done
done

printf 'CURRENT_SOURCE_CUDA_SELECTOR_AND_DYNAMIC_CONTROLS_PASSED\n' \
  > "${evidence_root}/verdict.txt"

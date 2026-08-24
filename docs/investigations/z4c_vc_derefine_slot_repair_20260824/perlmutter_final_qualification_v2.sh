#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?detached AthenaK source at the qualified revision}"
: "${CPU_BUILD_ROOT:?Perlmutter CPU MPI/OpenMP build root}"
: "${CUDA_BUILD_ROOT:?Perlmutter CUDA MPI build root}"
: "${OUTPUT_ROOT:?fresh qualification output directory}"

expected_source=6dd20656a305f2543bbbd7001550c6ac67019180
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 4
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${expected_source}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1)"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=4 KOKKOS_NUM_THREADS=4
export SLURM_MPI_TYPE=cray_shasta

{
  scontrol show job "${SLURM_JOB_ID}"
  module list
  git -C "${SOURCE_ROOT}" rev-parse HEAD HEAD^{tree}
  git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD
  sha256sum "${CPU_BUILD_ROOT}/src/athena" "${CPU_BUILD_ROOT}/CMakeCache.txt"
  sha256sum "${CUDA_BUILD_ROOT}/src/athena" "${CUDA_BUILD_ROOT}/CMakeCache.txt"
  "${python_bin}" --version
  nvidia-smi -L
} > "${OUTPUT_ROOT}/provenance.txt" 2>&1

export MPICH_GPU_SUPPORT_ENABLED=0
set +e
ctest --test-dir "${CPU_BUILD_ROOT}" --output-on-failure -j1 \
  > "${OUTPUT_ROOT}/cpu_full.log" 2>&1
cpu_status=$?
set -e

run_generated_cuda_test() {
  local index=$1
  local command
  command=$(ctest --test-dir "${CUDA_BUILD_ROOT}" -N -V -I "${index},${index}" |
    sed -n 's/^[0-9][0-9]*: Test command: //p')
  test -n "${command}"
  command=${command/\/usr\/bin\/python3/${python_bin}}
  printf '%s\n' "${command}" > "${OUTPUT_ROOT}/cuda_test_${index}.command"
  eval "${command}" > "${OUTPUT_ROOT}/cuda_test_${index}.log" 2>&1
}

export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
cuda_amr_status=0
for index in $(seq 25 64); do
  if ! run_generated_cuda_test "${index}"; then
    cuda_amr_status=1
  fi
done

set +e
ctest --test-dir "${CUDA_BUILD_ROOT}" --output-on-failure -I 155,156 -j1 \
  > "${OUTPUT_ROOT}/cuda_kernel.log" 2>&1
cuda_kernel_status=$?
set -e

{
  printf 'cpu_full=%d\n' "${cpu_status}"
  printf 'cuda_amr_25_64=%d\n' "${cuda_amr_status}"
  printf 'cuda_kernel_155_156=%d\n' "${cuda_kernel_status}"
} > "${OUTPUT_ROOT}/status.txt"

find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${OUTPUT_ROOT}/SHA256SUMS"
(cd "${OUTPUT_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)

test "${cpu_status}" -eq 0
test "${cuda_amr_status}" -eq 0
test "${cuda_kernel_status}" -eq 0

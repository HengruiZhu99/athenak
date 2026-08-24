#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?detached AthenaK source at the qualified revision}"
: "${CPU_BUILD_ROOT:?Perlmutter CPU MPI/OpenMP build root}"
: "${CUDA_BUILD_ROOT:?Perlmutter CUDA MPI build root}"
: "${OUTPUT_ROOT:?fresh targeted qualification output directory}"

expected_source=6dd20656a305f2543bbbd7001550c6ac67019180
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 4
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${expected_source}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1)"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export SLURM_MPI_TYPE=cray_shasta

{
  scontrol show job "${SLURM_JOB_ID}"
  module list
  git -C "${SOURCE_ROOT}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD
  sha256sum "${CPU_BUILD_ROOT}/src/athena" "${CPU_BUILD_ROOT}/CMakeCache.txt"
  sha256sum "${CUDA_BUILD_ROOT}/src/athena" "${CUDA_BUILD_ROOT}/CMakeCache.txt"
  "${python_bin}" --version
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
} > "${OUTPUT_ROOT}/provenance.txt" 2>&1

run_generated_test() {
  local build_root=$1
  local index=$2
  local prefix=$3
  local command
  command=$(ctest --test-dir "${build_root}" -N -V -I "${index},${index}" |
    sed -n 's/^[0-9][0-9]*: Test command: //p')
  test -n "${command}"
  command=${command/\/usr\/bin\/python3/${python_bin}}
  printf '%s\n' "${command}" > "${OUTPUT_ROOT}/${prefix}_${index}.command"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader \
    > "${OUTPUT_ROOT}/${prefix}_${index}.gpu_before"
  set +e
  eval "${command}" > "${OUTPUT_ROOT}/${prefix}_${index}.log" 2>&1
  local status=$?
  set -e
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader \
    > "${OUTPUT_ROOT}/${prefix}_${index}.gpu_after"
  printf '%d\n' "${status}" > "${OUTPUT_ROOT}/${prefix}_${index}.status"
  return "${status}"
}

status=0

export MPICH_GPU_SUPPORT_ENABLED=0
if ! run_generated_test "${CPU_BUILD_ROOT}" 58 cpu; then
  status=1
fi

export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU

# The narrow two-dimensional and writer cases exercise the repaired production
# path without the large memory footprint of the unrelated broad matrix.
for index in 25 26 27 28 29 30 34; do
  if ! run_generated_test "${CUDA_BUILD_ROOT}" "${index}" cuda; then
    status=1
  fi
done

# Cover all same-rank, move-left, move-right, mixed, and dual split-family
# ownership layouts on CUDA with MPI rank redistribution.
for index in 52 53 54 55 56 57 58; do
  if ! run_generated_test "${CUDA_BUILD_ROOT}" "${index}" cuda; then
    status=1
  fi
done

# Run the three required three-dimensional O2/O4/O6 cases separately so each
# releases its allocation before the next starts.
for index in 31 32 33; do
  if ! run_generated_test "${CUDA_BUILD_ROOT}" "${index}" cuda3d; then
    status=1
  fi
done

for index in 155 156; do
  if ! run_generated_test "${CUDA_BUILD_ROOT}" "${index}" kernel; then
    status=1
  fi
done

printf 'overall=%d\n' "${status}" > "${OUTPUT_ROOT}/status.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${OUTPUT_ROOT}/SHA256SUMS"
(cd "${OUTPUT_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
exit "${status}"

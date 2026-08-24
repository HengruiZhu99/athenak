#!/usr/bin/env bash
set -euo pipefail

: "${CPU_BUILD_ROOT:?Perlmutter CPU MPI/OpenMP build root}"
: "${CUDA_BUILD_ROOT:?Perlmutter CUDA MPI build root}"
: "${OUTPUT_ROOT:?fresh qualification output directory}"

expected_source=d2596707e808aea7ec6167df937d71dc4dbe429e
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 4
test "$(git -C "${CPU_BUILD_ROOT}/../source" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${CUDA_BUILD_ROOT}/../source" rev-parse HEAD)" = "${expected_source}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=4 KOKKOS_NUM_THREADS=4
export SLURM_MPI_TYPE=cray_shasta

{
  scontrol show job "${SLURM_JOB_ID}"
  module list
  sha256sum "${CPU_BUILD_ROOT}/src/athena" "${CPU_BUILD_ROOT}/CMakeCache.txt"
  sha256sum "${CUDA_BUILD_ROOT}/src/athena" "${CUDA_BUILD_ROOT}/CMakeCache.txt"
  "${python_bin}" --version
} > "${OUTPUT_ROOT}/provenance.txt" 2>&1

# These four tests explicitly launch two- and four-rank srun steps.  The outer
# allocation therefore reserves four tasks, but no enclosing srun is used.
export MPICH_GPU_SUPPORT_ENABLED=0
set +e
ctest --test-dir "${CPU_BUILD_ROOT}" --output-on-failure -I 51,54 -j1 \
  > "${OUTPUT_ROOT}/cpu_mpi.log" 2>&1
cpu_status=$?
set -e

# The CUDA build was configured on a node whose /usr/bin/python3 is too old for
# the test harness.  Preserve each CTest-generated command exactly, replacing
# only that interpreter with the supported NERSC Python 3.11.
run_generated_python_test() {
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
cuda_python_status=0
for index in 26 27 28 29 30 31 32 33 46 47 48; do
  if ! run_generated_python_test "${index}"; then
    cuda_python_status=1
  fi
done

set +e
ctest --test-dir "${CUDA_BUILD_ROOT}" --output-on-failure -I 151,152 -j1 \
  > "${OUTPUT_ROOT}/cuda_kernel.log" 2>&1
cuda_kernel_status=$?
set -e

{
  printf 'cpu_mpi=%d\n' "${cpu_status}"
  printf 'cuda_python=%d\n' "${cuda_python_status}"
  printf 'cuda_kernel=%d\n' "${cuda_kernel_status}"
} > "${OUTPUT_ROOT}/status.txt"

find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${OUTPUT_ROOT}/SHA256SUMS"
(cd "${OUTPUT_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)

test "${cpu_status}" -eq 0
test "${cuda_python_status}" -eq 0
test "${cuda_kernel_status}" -eq 0

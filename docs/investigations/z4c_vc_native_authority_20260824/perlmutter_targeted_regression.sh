#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_ROOT:?fresh evidence directory}"

source_root=/pscratch/sd/h/hzhu/z4c-vc-derefine-slot-repair-20260824/source
build_root=/pscratch/sd/h/hzhu/z4c-vc-native-authority-20260824/build-cuda-mpi-deterministic-history
expected_source=d63519328214a6315a9cc1f7d5e4a1aa4bca21b0
expected_tree=9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 4
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta

{
  scontrol show job "${SLURM_JOB_ID}"
  module list
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt"
  "${python_bin}" --version
  nvidia-smi -L
} > "${OUTPUT_ROOT}/provenance.txt" 2>&1

status=0
for index in 25 26 27 28 29 30 31 32 33 34 52 53 54 55 56 57 58 150 156 157; do
  command=$(ctest --test-dir "${build_root}" -N -V -I "${index},${index}" |
    sed -n 's/^[0-9][0-9]*: Test command: //p')
  test -n "${command}"
  command=${command/\/usr\/bin\/python3/${python_bin}}
  printf '%s\n' "${command}" > "${OUTPUT_ROOT}/test_${index}.command"
  set +e
  eval "${command}" > "${OUTPUT_ROOT}/test_${index}.log" 2>&1
  test_status=$?
  set -e
  printf '%d\n' "${test_status}" > "${OUTPUT_ROOT}/test_${index}.status"
  if test "${test_status}" -ne 0; then
    status=1
  fi
done

printf 'overall=%d\n' "${status}" > "${OUTPUT_ROOT}/status.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${OUTPUT_ROOT}/SHA256SUMS"
(cd "${OUTPUT_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
exit "${status}"

#!/bin/bash -l
# Configure and compile on an Aurora login/head node. Runtime qualification is
# deliberately deferred to aurora_qualify_run.pbs on a PVC tile.
set -euo pipefail
: "${ATHENAK_ROOT:?exact clean AthenaK checkout required}"
: "${ATHENAK_EXPECTED_COMMIT:?exact AthenaK commit required}"
: "${IRIS_ROOT:?exact clean IrisK checkout required}"
: "${AURORA_PHASE_ROOT:?fresh build output root required}"
if [[ -e "${AURORA_PHASE_ROOT}" ]]; then
  printf 'refusing to reuse build root: %s\n' "${AURORA_PHASE_ROOT}" >&2
  exit 2
fi
mkdir -p "${AURORA_PHASE_ROOT}"
phase_root="$(cd "${AURORA_PHASE_ROOT}" && pwd)"
athena_root="$(cd "${ATHENAK_ROOT}" && pwd)"
iris_root="$(cd "${IRIS_ROOT}" && pwd)"
iris_build="${phase_root}/iris-build"
athena_build="${phase_root}/athena-build"

module purge
module load oneapi/release/2025.3.1 gcc/13.4.0 cmake/3.31.11 python/3.12.12
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1

test "$(git -C "${athena_root}" rev-parse HEAD)" = "${ATHENAK_EXPECTED_COMMIT}"
test -z "$(git -C "${athena_root}" status --porcelain)"
test "$(git -C "${iris_root}" rev-parse HEAD)" = 2a069fd0497ef4352d4ecd28c6879ac47b84a5a1
test -z "$(git -C "${iris_root}" status --porcelain)"
{
  date --iso-8601=seconds
  hostname
  git -C "${athena_root}" rev-parse HEAD HEAD^{tree}
  git -C "${athena_root}" submodule status
  git -C "${iris_root}" rev-parse HEAD HEAD^{tree}
  git -C "${iris_root}" submodule status
  module list
  command -v icpx
  icpx --version | head -n 1
  cmake --version | head -n 1
} > "${phase_root}/provenance.log" 2>&1

cmake -S "${iris_root}" -B "${iris_build}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DIRIS_ENABLE_MPI=OFF -DIRIS_USE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SYCL=OFF -DBUILD_TESTING=OFF > "${phase_root}/iris-configure.log" 2>&1
cmake --build "${iris_build}" --parallel 64 --target iris_athenak_interpolator > "${phase_root}/iris-build.log" 2>&1
test -f "${iris_build}/src/libiris_athenak_interpolator.a"

cmake -S "${athena_root}" -B "${athena_build}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DPROBLEM=z4c_irisk_xcts -DIRISK_ROOT="${iris_root}" -DIRISK_INTERPOLATOR_LIBRARY="${iris_build}/src/libiris_athenak_interpolator.a" -DBUILD_TESTING=ON -DAthena_BUILD_UNIT_TESTS=ON -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON -DKokkos_ENABLE_TESTS=OFF > "${phase_root}/athena-configure.log" 2>&1
cmake --build "${athena_build}" --parallel 64 --target athena athena_z4c_state_admissibility_unit_test athena_z4c_amr_jump_diagnostic_unit_test athena_z4c_cartoon_amr_transfer_qualification_unit_test > "${phase_root}/athena-build.log" 2>&1
grep -qx 'Kokkos_ENABLE_SYCL:BOOL=ON' "${athena_build}/CMakeCache.txt"
grep -qx 'Kokkos_ARCH_INTEL_PVC:BOOL=ON' "${athena_build}/CMakeCache.txt"
grep -qx 'Athena_ENABLE_MPI:BOOL=OFF' "${athena_build}/CMakeCache.txt"
test -x "${athena_build}/src/athena"
sha256sum "${iris_build}/src/libiris_athenak_interpolator.a" "${athena_build}/src/athena" "${phase_root}"/*.log > "${phase_root}/SHA256SUMS"
printf 'AURORA_N256_SCRATCH_HEAD_BUILD_PASS\n' > "${phase_root}/status"

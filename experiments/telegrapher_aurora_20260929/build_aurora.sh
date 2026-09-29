#!/bin/bash -l
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
build="$root/build/aurora-sycl"
module load cmake
# Kokkos 4.7 compatibility with oneAPI 2026.1; retain the pinned submodule.
python3 - <<'PATCH'
from pathlib import Path
p = Path('kokkos/core/src/setup/Kokkos_Setup_SYCL.hpp')
s = p.read_text()
s = s.replace('#error SYCL_EXT_INTEL_USM_ADDRESS_SPACES undefined!',
'''template <typename T>
using sycl_device_ptr = sycl::global_ptr<T>;
template <typename T>
using sycl_host_ptr = sycl::global_ptr<T>;''')
if s != p.read_text():
    p.write_text(s)
# oneAPI 2026.1 forwards Kokkos 4.7's escaped, space-containing backend
# argument as a single ocloc token. Repeated flags preserve separate tokens.
p = Path('kokkos/cmake/kokkos_arch.cmake')
s = p.read_text().replace(
    'set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device 12.60.7")',
    'set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend -device '
    '-Xsycl-target-backend=spir64_gen 12.60.7)')
s = s.replace('-Xsycl-target-backend -device -Xsycl-target-backend 12.60.7',
              '-Xsycl-target-backend -device -Xsycl-target-backend=spir64_gen 12.60.7')
if s != p.read_text():
    p.write_text(s)
PATCH
mkdir -p "$build"
module list > "$build/modules.txt" 2>&1
icpx --version > "$build/compiler.txt"
git rev-parse HEAD > "$build/source_commit.txt"
git -C kokkos diff > "$build/kokkos-compat.patch"
cmake -U 'KOKKOS_*_OPTIONS_CHECK' -S "$root" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$root/experiments/telegrapher_aurora_20260929/sycl_compile.py" \
  -DMPI_CXX_COMPILER=mpicxx \
  -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_OPENMP=ON \
  -DOpenMP_CXX_FLAGS=-fiopenmp -DOpenMP_CXX_LIB_NAMES='iomp5;pthread' \
  -DPROBLEM=built_in_pgens \
  -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON \
  -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON \
  -DCMAKE_CXX_FLAGS="-O3 -fsycl -fiopenmp" \
  -DCMAKE_C_FLAGS="-O3 -ffp-model=precise" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsycl-max-parallel-link-jobs=8"
cmake --build "$build" -j "${BUILD_JOBS:-8}"
sha256sum "$build/src/athena" > "$build/executable.sha256"

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
p.write_text(s)
PATCH
mkdir -p "$build"
module list > "$build/modules.txt" 2>&1
icpx --version > "$build/compiler.txt"
git rev-parse HEAD > "$build/source_commit.txt"
git -C kokkos diff > "$build/kokkos-compat.patch"
cmake -S "$root" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=icx \
  -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_OPENMP=ON \
  -DPROBLEM=built_in_pgens \
  -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON \
  -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON \
  -DCMAKE_CXX_FLAGS="-O3 -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen -device\ pvc" \
  -DCMAKE_C_FLAGS="-O3 -ffp-model=precise"
cmake --build "$build" -j "${BUILD_JOBS:-8}"
sha256sum "$build/src/athena" > "$build/executable.sha256"

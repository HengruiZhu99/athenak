#!/bin/bash -l
# Build only. Never submits or edits another campaign's files.
set -euo pipefail
root=${1:?Pass the dedicated scratch task directory}
cd "$root"
/usr/bin/python3.11 source/tst/hyperbolic_shift_3d/aurora/refined/storage.py "$root" 8000000000 > build-storage-before.json
module load oneapi/release/2025.3.1
module list > build-modules.txt 2>&1
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int' \
  -DCMAKE_CXX_FLAGS_RELEASE=-O2 -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
  -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON -DKokkos_ENABLE_SYCL=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_PVC=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  > configure.log 2>&1
cd build
make -j64 hyperbolic_shift_3d_refined hyperbolic_shift_3d_refined_amr \
  hyperbolic_shift_3d_refined_profile hyperbolic_shift_3d_refined_device \
  hyperbolic_shift_3d_amr_device_mapping > ../build.log 2>&1
cd ..
sha256sum build/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined* \
  build/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_device_mapping > binaries.sha256
/usr/bin/python3.11 source/tst/hyperbolic_shift_3d/aurora/refined/storage.py "$root" > build-storage-after.json

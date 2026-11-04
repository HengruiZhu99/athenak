#!/bin/bash
set -euo pipefail
cd /home/hzhu/hyperboloidal-performance-scaling-20261103
module load oneapi/release/2025.3.1
cp -a control-source-v2 m1-source
tar -xzf m1-overlay.tar.gz -C m1-source
cmake -S m1-source -B m1-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int' -DCMAKE_CXX_FLAGS_RELEASE=-O2 -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_PVC=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > evidence/m1-configure.log 2>&1
cmake --build m1-build --target hyperbolic_shift_3d_amr_evolution hyperbolic_shift_3d_amr_diagnostic_transfer hyperbolic_shift_3d_owned_padded_snapshot hyperbolic_shift_3d_amr_device_mapping hyperbolic_amr_allocation_tool -j32 > evidence/m1-build.log 2>&1
sha256sum m1-build/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr* > evidence/m1-binaries.sha256

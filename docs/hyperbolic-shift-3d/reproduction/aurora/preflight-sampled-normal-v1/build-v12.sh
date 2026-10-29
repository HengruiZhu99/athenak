#!/bin/bash
set -euo pipefail
cd /home/hzhu/hyperbolic-core-shell-20261028
module load oneapi/release/2025.3.1
module list > build-logs/v12/modules.txt 2>&1
icpx --version > build-logs/v12/compiler.txt 2>&1
mpicxx -show > build-logs/v12/mpi-wrapper.txt 2>&1
cmake -S source-v12 -B build-sycl-mpi-v12 \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
 '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int' \
 -DCMAKE_CXX_FLAGS_RELEASE=-O2 \
 -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
 -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON \
 -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON \
 -DKokkos_ARCH_INTEL_PVC=ON > build-logs/v12/configure.log 2>&1
make -C build-sycl-mpi-v12 -j64 \
 hyperbolic_shift_3d_mpi_sampled_normal \
 hyperbolic_shift_3d_mpi_dissipative \
 hyperbolic_shift_3d_sampled_normal_device \
 hyperbolic_shift_3d_composite_device hyperbolic_shift_3d_composite_exchange \
 hyperbolic_shift_3d_scri_device > build-logs/v12/build.log 2>&1

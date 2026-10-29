#!/bin/bash
set -euo pipefail
cd /home/hzhu/hyperbolic-core-shell-20261028
module load oneapi/release/2025.3.1
module list > build-logs/v11/modules.txt 2>&1
icpx --version > build-logs/v11/compiler.txt 2>&1
mpicxx -show > build-logs/v11/mpi-wrapper.txt 2>&1
cmake -S source-v11 -B build-sycl-mpi-v11 \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
 '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int' \
 -DCMAKE_CXX_FLAGS_RELEASE=-O2 \
 -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
 -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON \
 -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON \
 -DKokkos_ARCH_INTEL_PVC=ON > build-logs/v11/configure.log 2>&1
make -C build-sycl-mpi-v11 -j64 \
 hyperbolic_shift_3d_mpi_dissipative \
 hyperbolic_shift_3d_mpi_projected_angular \
 hyperbolic_shift_3d_composite_device hyperbolic_shift_3d_composite_exchange \
 hyperbolic_shift_3d_scri_device hyperbolic_shift_3d_scri_angular \
 hyperbolic_shift_3d_algebraic_projection hyperbolic_shift_3d_angular_dissipation \
 hyperbolic_shift_3d_scri_dissipation_closure > build-logs/v11/build.log 2>&1

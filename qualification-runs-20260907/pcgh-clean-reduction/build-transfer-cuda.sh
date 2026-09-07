#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-transfer-001
trap 'printf "%s\n" "$?" > build.exit' EXIT
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/source/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_AMPERE80=ON -DAthena_ENABLE_MPI=ON \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
  -DPROBLEM=../../analysis/pc_gh_clean_reduction/transfer_mesh_oracle \
  > configure.log 2>&1
cmake --build build -j4 > build.log 2>&1

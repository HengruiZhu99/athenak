#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-legacy
for arm in collision current; do
  mkdir "$arm-source"
  tar -xzf "$arm-legacy-source.tar.gz" -C "$arm-source"
  cmake -S "$arm-source" -B "build-$arm" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$PWD/$arm-source/kokkos/bin/nvcc_wrapper" -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON -DAthena_ENABLE_MPI=OFF -DPROBLEM=../../analysis/pc_gh_clean_reduction/legacy_oracle > "configure-$arm.log" 2>&1
  cmake --build "build-$arm" -j4 > "build-$arm.log" 2>&1
done

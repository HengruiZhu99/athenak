#!/bin/bash
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper"
  -DAthena_ENABLE_MPI=ON -DAthena_VC_DEVICE_MPI=ON -DMPI_CXX_COMPILER="$(command -v mpicxx)"
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL100=ON -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF)
cmake -S . -B build "${flags[@]}"
cmake --build build --target athena --parallel "$SLURM_CPUS_PER_TASK"
cmake -S . -B build-check -DAthena_BUILD_UNIT_TESTS=ON "${flags[@]}"
cmake --build build-check --target athena_boltzmann_fourth_accuracy --parallel "$SLURM_CPUS_PER_TASK"
sha256sum build/src/athena build-check/athena_boltzmann_fourth_accuracy
build-check/athena_boltzmann_fourth_accuracy

#!/bin/bash
# Submit from the staged checkout in a StellarAI single-GPU Slurm allocation.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
module list
hostname
nvidia-smi -L
nvcc --version
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=OFF -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_BLACKWELL100=ON -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF
cmake --build build --target athena --parallel "$SLURM_CPUS_PER_TASK"
sha256sum build/src/athena

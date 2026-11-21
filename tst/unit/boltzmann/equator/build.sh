#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --job-name=eq-build
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=ON -DAthena_VC_DEVICE_MPI=ON -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL100=ON -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF
cmake --build build --target athena --parallel "$SLURM_CPUS_PER_TASK"
sha256sum build/src/athena

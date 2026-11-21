#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:12:00
#SBATCH --job-name=eq-accuracy
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
cmake -S . -B build-accuracy -DCMAKE_BUILD_TYPE=Release -DAthena_BUILD_UNIT_TESTS=ON \
    -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" -DAthena_ENABLE_MPI=ON \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_BLACKWELL100=ON -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF
cmake --build build-accuracy --target athena_boltzmann_fourth_amr --parallel "$SLURM_CPUS_PER_TASK"
build-accuracy/athena_boltzmann_fourth_amr

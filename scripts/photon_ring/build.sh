#!/usr/bin/env bash
# Pass toolchain, backend and architecture choices as ordinary CMake arguments.
set -euo pipefail
build_dir=${1:?usage: build.sh BUILD_DIR [CMake options...]}
shift
src=$(cd "$(dirname "$0")/../.." && pwd)
cmake -S "$src" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DAthena_ENABLE_MPI=ON "$@"
cmake --build "$build_dir" --target athena --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-${SLURM_CPUS_PER_TASK:-2}}"

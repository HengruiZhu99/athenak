#!/bin/bash -l
# Build a private frozen source snapshot; never reuse or mutate campaign builds.
set -euo pipefail
root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-regularity-20261110
source_dir=$root/source-v1
build_dir=$root/build-v1
[[ -f "$source_dir/src/z4c_hyperboloidal/scri_reconstruction.hpp" ]]
[[ ! -e "$build_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$build_dir"
module list > "$build_dir/modules.txt" 2>&1
icpx --version > "$build_dir/compiler.txt"
mpicxx -show > "$build_dir/mpi-wrapper.txt"
python3 "$source_dir/tst/z4c_hyperboloidal/aurora/source_manifest.py" "$source_dir" "$build_dir/source-sha256.json"
cmake -S "$source_dir" -B "$build_dir" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
 '-DCMAKE_CXX_FLAGS=-fp-model=precise -ffp-contract=off -fno-sycl-id-queries-fit-in-int' \
 -DCMAKE_CXX_FLAGS_RELEASE=-O2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF -DAthena_ENABLE_OPENMP=OFF \
 -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_PVC=ON \
 -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF -DKokkos_ENABLE_IMPL_SYCL_OUT_OF_ORDER_QUEUES=OFF \
 > "$build_dir/configure.log" 2>&1
make -C "$build_dir" -j64 > "$build_dir/build.log" 2>&1
sha256sum "$build_dir/src/athena" > "$build_dir/executable.sha256"
python3 "$source_dir/tst/z4c_hyperboloidal/aurora/source_manifest.py" "$source_dir" "$build_dir/source-sha256.json" --check > "$build_dir/source-check.txt"
printf 'Experimental build complete: %s\n' "$build_dir"

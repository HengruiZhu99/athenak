#!/usr/bin/env bash
set -euo pipefail

root=/tmp/hzhu-athenak-history-extrema-homebuild-20260901
source_root=${root}/source
build_root=${root}/build
irisk_root=${root}/irisk-authority
campaign=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901

module load PrgEnv-gnu cudatoolkit cmake cray-hdf5

# The local worktree submodule marker is meaningless after the source is
# packaged onto Perlmutter.  Kokkos configures correctly as a source archive
# when that temporary marker is absent.
if [[ -f ${source_root}/kokkos/.git ]]; then
  rm -- "${source_root}/kokkos/.git"
fi
if [[ -d ${build_root} ]]; then
  rm -rf -- "${build_root}"
fi

date -Is
hostname
cmake -S "${source_root}" -B "${build_root}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.36/bin/CC \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_IRISK_INTERPOLATOR=ON \
  -DIRISK_ROOT="${irisk_root}" \
  -DIRISK_INTERPOLATOR_LIBRARY="${irisk_root}/build/serial-gcc/src/libiris_athenak_interpolator.a" \
  -DAthena_BUILD_UNIT_TESTS=ON
cmake --build "${build_root}" --target athena -j8
sha256sum "${source_root}/src/outputs/history.cpp" "${build_root}/src/athena"
mkdir -p "${campaign}/build"
cp -p "${build_root}/src/athena" "${campaign}/build/athena.history_extrema"
sha256sum "${campaign}/build/athena.history_extrema"
date -Is

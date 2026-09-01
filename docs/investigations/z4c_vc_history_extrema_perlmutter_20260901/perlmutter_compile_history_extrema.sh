#!/usr/bin/env bash
set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901
source_root=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/supercritical-horizon-20260830/source
build_root=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/supercritical-horizon-20260830/build-cuda-mpi
athena=${build_root}/src/athena
temporary_root=/tmp/hzhu-athenak-history-extrema-20260901

module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
mkdir -p "${campaign}/build" "${temporary_root}"
date -Is
hostname

flags_file=${build_root}/src/CMakeFiles/athena.dir/flags.make
defines=$(sed -n 's/^CXX_DEFINES = //p' "${flags_file}")
includes=$(sed -n 's/^CXX_INCLUDES = //p' "${flags_file}")
cxx_flags=$(sed -n 's/^CXX_FLAGS = //p' "${flags_file}")
temporary_object=${temporary_root}/history.cpp.o
temporary_dependency=${temporary_root}/history.cpp.o.d
compile_command="${source_root}/kokkos/bin/kokkos_launch_compiler ${source_root}/kokkos/bin/nvcc_wrapper /opt/cray/pe/craype/2.7.36/bin/CC /opt/cray/pe/craype/2.7.36/bin/CC ${defines} ${includes} ${cxx_flags} -MD -MT src/CMakeFiles/athena.dir/outputs/history.cpp.o -MF ${temporary_dependency} -o ${temporary_object} -c ${source_root}/src/outputs/history.cpp"
echo HISTORY_EXTREMA_COMPILE_BEGIN
eval "${compile_command}"
echo HISTORY_EXTREMA_COMPILE_OK
sha256sum "${temporary_object}" "${temporary_dependency}"

object=${build_root}/src/CMakeFiles/athena.dir/outputs/history.cpp.o
dependency=${build_root}/src/CMakeFiles/athena.dir/outputs/history.cpp.o.d
cp -p "${object}" "${campaign}/build/history.cpp.o.pre_min_lapse"
cp -p "${dependency}" "${campaign}/build/history.cpp.o.pre_min_lapse.d"
cp -p "${temporary_object}" "${object}.new"
cp -p "${temporary_dependency}" "${dependency}.new"
mv "${object}.new" "${object}"
mv "${dependency}.new" "${dependency}"

cd "${build_root}/src"
cp -p "${athena}" "${campaign}/build/athena.pre_min_lapse"
bash CMakeFiles/athena.dir/link.txt
cp -p "${athena}" "${campaign}/build/athena.history_extrema"
sha256sum "${source_root}/src/outputs/history.cpp" "${object}" "${athena}" \
  "${campaign}/build/athena.history_extrema" | tee "${campaign}/build/SHA256SUMS"
date -Is

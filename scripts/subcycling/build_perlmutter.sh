#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
cmake -S "$root/source" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=CC -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_IRISK_INTERPOLATOR=ON -DAthena_BUILD_UNIT_TESTS=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_SERIAL=ON -DIRISK_ROOT=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823/source/irisk-authority -DIRISK_INTERPOLATOR_LIBRARY=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823/source/irisk-authority/build/serial-gcc/src/libiris_athenak_interpolator.a
cmake --build "$root/build" --target athena -j 8
sha256sum "$root/build/src/athena" > "$root/executable.sha256"

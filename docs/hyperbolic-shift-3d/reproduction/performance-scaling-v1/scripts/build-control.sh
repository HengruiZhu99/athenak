#!/bin/bash
set -euo pipefail
cd /home/hzhu/hyperboloidal-performance-scaling-20261103
mkdir -p evidence
module load oneapi/release/2025.3.1
module list > evidence/modules.txt 2>&1
icpx --version > evidence/compiler.txt
mpicxx -show > evidence/mpi-wrapper.txt
qstat -u hzhu > evidence/jobs-initial.txt
qstat -Qf debug > evidence/queue-debug.txt
qstat -Qf debug-scaling > evidence/queue-debug-scaling.txt
sbank list allocations -p CompactBinaryMerger > evidence/account.txt
lfs quota -u hzhu /home > evidence/quota.txt
mkdir control-source-v2
tar -xzf control-source-v2.tar.gz -C control-source-v2
cmake -S control-source-v2 -B control-build-v2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int' -DCMAKE_CXX_FLAGS_RELEASE=-O2 -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_PVC=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > evidence/control-configure-v2.log 2>&1
cmake --build control-build-v2 --target hyperbolic_shift_3d_amr_evolution hyperbolic_shift_3d_amr_transfer hyperbolic_shift_3d_amr_operators -j32 > evidence/control-build-v2.log 2>&1
sha256sum control-build-v2/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr* > evidence/control-binaries.sha256

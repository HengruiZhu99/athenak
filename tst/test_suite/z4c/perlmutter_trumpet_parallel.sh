#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?allocated compute node required}"
PUNCTURE_ROOT=/pscratch/sd/h/hzhu/codex-hispid-trumpet-20261005
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=cores
printf '%s\n' "$SLURM_JOB_ID" > "$PUNCTURE_ROOT/parallel-consumer-job.txt"
cmake -S "$PUNCTURE_ROOT/athenak-parallel" -B "$PUNCTURE_ROOT/build-athenak-openmp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/cray/pe/gcc-native/14/bin/gcc -DCMAKE_CXX_COMPILER=/opt/cray/pe/gcc-native/14/bin/g++ -DPROBLEM=z4c/hispid -DHISPID_ROOT="$PUNCTURE_ROOT/source" -DHISPID_LIBRARY_DIR="$PUNCTURE_ROOT/build-sampler" -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=OFF -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=ON > "$PUNCTURE_ROOT/configure-athenak-openmp.log" 2>&1
/usr/bin/time -v cmake --build "$PUNCTURE_ROOT/build-athenak-openmp" -j8 > "$PUNCTURE_ROOT/build-athenak-openmp.log" 2>&1
sha256sum "$PUNCTURE_ROOT/build-athenak-openmp/src/athena" > "$PUNCTURE_ROOT/image-athenak-openmp.sha256"
python3 "$PUNCTURE_ROOT/athenak-parallel/tst/test_suite/z4c/check_hispid_parallel_geometry.py" --executable "$PUNCTURE_ROOT/build-athenak-openmp/src/athena" --baseline "$PUNCTURE_ROOT/first-physics/horizons/controls.json" --output "$PUNCTURE_ROOT/parallel-consumer-control" --threads 16 > "$PUNCTURE_ROOT/parallel-consumer-control.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?compute allocation required}"
PUNCTURE_ROOT=/pscratch/sd/h/hzhu/codex-hispid-trumpet-20261005
PUNCTURE_RUN="$PUNCTURE_ROOT/convergence/consumer"
mkdir -p "$PUNCTURE_RUN"
printf '%s\n' "$SLURM_JOB_ID" > "$PUNCTURE_RUN/job-id.txt"
cmake -S "$PUNCTURE_ROOT/athenak-polar" -B "$PUNCTURE_ROOT/build-athenak-polar" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/cray/pe/gcc-native/14/bin/gcc -DCMAKE_CXX_COMPILER=/opt/cray/pe/gcc-native/14/bin/g++ -DPROBLEM=z4c/hispid -DHISPID_ROOT="$PUNCTURE_ROOT/source-polar" -DHISPID_LIBRARY_DIR="$PUNCTURE_ROOT/build-polar-sampler" -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=OFF -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=ON > "$PUNCTURE_RUN/configure.log" 2>&1
/usr/bin/time -v cmake --build "$PUNCTURE_ROOT/build-athenak-polar" -j8 > "$PUNCTURE_RUN/build.log" 2>&1
sha256sum "$PUNCTURE_ROOT/build-athenak-polar/src/athena" "$PUNCTURE_ROOT/build-polar-sampler/libHiSpID.so" "$PUNCTURE_ROOT/athenak-polar/src/pgen/z4c/hispid.cpp" "$PUNCTURE_ROOT/athenak-polar/src/pgen/z4c/hispid_checkpoint.hpp" "$PUNCTURE_ROOT/athenak-polar/src/z4c/fastflow.cpp" "$PUNCTURE_ROOT/athenak-polar/src/z4c/fastflow.hpp" > "$PUNCTURE_RUN/images-and-sources.sha256"
printf 'completed\n' > "$PUNCTURE_RUN/completed.txt"

#!/bin/bash -l
# Prepared build command; this script has not been executed on Aurora.
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'Usage: %s SOURCE_DIR NEW_BUILD_DIR\n' "$0" >&2
  exit 2
fi
source_dir=$(realpath "$1")
build_dir=$(realpath -m "$2")
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
case "$source_dir" in "$task_root"/source) ;; *) exit 2 ;; esac
case "$build_dir" in "$task_root"/build-*) ;; *) exit 2 ;; esac
[[ ! -e "$build_dir" ]] || { printf 'Build directory already exists\n' >&2; exit 2; }
[[ -f "$source_dir/src/z4c_hyperboloidal/horizon_sampling.cpp" ]]
module load oneapi/release/2025.3.1
mkdir "$build_dir"
module list > "$build_dir/modules.txt" 2>&1
icpx --version > "$build_dir/compiler.txt"
mpicxx -show > "$build_dir/mpi-wrapper.txt"
if [[ -f "$source_dir/source-revision.json" ]]; then
  cp "$source_dir/source-revision.json" "$build_dir/source-revision.json"
else
  git -C "$source_dir" rev-parse HEAD > "$build_dir/source-commit.txt"
  git -C "$source_dir" status --porcelain > "$build_dir/source-status.txt"
  git -C "$source_dir/kokkos" rev-parse HEAD > "$build_dir/kokkos-commit.txt"
fi
/usr/bin/python3.11 "$source_dir/tst/z4c_hyperboloidal/aurora/source_manifest.py" \
  "$source_dir" "$build_dir/source-sha256.json"
cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  '-DCMAKE_CXX_FLAGS=-fp-model=precise -ffp-contract=off -fno-sycl-id-queries-fit-in-int' \
  -DCMAKE_CXX_FLAGS_RELEASE=-O2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
  -DAthena_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SYCL=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_PVC=ON \
  -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF \
  -DKokkos_ENABLE_IMPL_SYCL_OUT_OF_ORDER_QUEUES=OFF \
  > "$build_dir/configure.log" 2>&1
make -C "$build_dir" -j64 > "$build_dir/build.log" 2>&1
/usr/bin/python3.11 - "$build_dir" <<'PY'
import json,pathlib,sys
build=pathlib.Path(sys.argv[1])
config=(build/'config.hpp').read_text()
assert '#define MPI_PARALLEL_ENABLED 1' in config
assert '#define SINGLE_PRECISION_ENABLED 0' in config
commands=json.loads((build/'compile_commands.json').read_text())
native=[x['command'] for x in commands if '/src/z4c_hyperboloidal/' in x['file']]
assert native, 'native module absent from build'
for command in native:
    for flag in ['-ffp-contract=off','-fp-model=precise','-O2',
                 '-fsycl','spir64_gen','12.60.7']:
        assert flag in command, (flag,command)
print('Native MPI float64 SYCL/PVC build flags verified')
PY
sha256sum "$build_dir/src/athena" > "$build_dir/executable.sha256"
/usr/bin/python3.11 "$source_dir/tst/z4c_hyperboloidal/aurora/source_manifest.py" \
  "$source_dir" "$build_dir/source-sha256.json" --check > "$build_dir/source-check.txt"
printf 'Built original native executable: %s/src/athena\n' "$build_dir"

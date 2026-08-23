#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
source_root=${campaign_root}/source/athenak
build_root=${campaign_root}/build/phase0-host-base
evidence_root=${campaign_root}/evidence/phase0-host-base
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
expected_source=2d59f85c11cb0da4614c84a695d64f032fb9eec7
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test ! -e "${build_root}"
test ! -e "${evidence_root}"
mkdir -p "${build_root}" "${evidence_root}"

finish() {
  status=$?
  set +e
  printf '%s\n' "${status}" > "${evidence_root}/exit-status.txt"
  find "${evidence_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 -r sha256sum > "${evidence_root}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=false
export KOKKOS_NUM_THREADS=8

scontrol show job "${SLURM_JOB_ID}" > "${evidence_root}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
env | sort > "${evidence_root}/environment.txt"
module -t list > "${evidence_root}/modules.txt" 2>&1
CC --version > "${evidence_root}/compiler.txt" 2>&1
cmake --version > "${evidence_root}/cmake.txt" 2>&1
"${python_bin}" --version > "${evidence_root}/python.txt" 2>&1
git -C "${source_root}" status --short --branch > "${evidence_root}/git-status.txt"
git -C "${source_root}" submodule status > "${evidence_root}/submodules.txt"
printf '%s\n' "${expected_source}" > "${evidence_root}/source-commit.txt"

configure=(
  cmake -S "${source_root}" -B "${build_root}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=cc
  -DCMAKE_CXX_COMPILER=CC
  -DPROBLEM=built_in_pgens
  -DBUILD_TESTING=ON
  -DAthena_BUILD_UNIT_TESTS=ON
  -DPYTHON_EXECUTABLE="${python_bin}"
  -DAthena_SINGLE_PRECISION=OFF
  -DAthena_ENABLE_MPI=OFF
  -DAthena_ENABLE_OPENMP=ON
  -DAthena_ENABLE_IRISK_INTERPOLATOR=OFF
  -DKokkos_ENABLE_TESTS=OFF
  -DKokkos_ENABLE_CUDA=OFF
  -DKokkos_ENABLE_OPENMP=ON
  -DKokkos_ENABLE_SERIAL=OFF
)
printf '%q ' "${configure[@]}" > "${evidence_root}/configure-command.txt"
printf '\n' >> "${evidence_root}/configure-command.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --exact \
  --kill-on-bad-exit=1 "${configure[@]}" \
  > "${evidence_root}/configure.log" 2> "${evidence_root}/configure.err"

build=(cmake --build "${build_root}" --parallel 32)
printf '%q ' "${build[@]}" > "${evidence_root}/build-command.txt"
printf '\n' >> "${evidence_root}/build-command.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --exact \
  --kill-on-bad-exit=1 "${build[@]}" \
  > "${evidence_root}/build.log" 2> "${evidence_root}/build.err"

athena=${build_root}/src/athena
test -x "${athena}"
sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
  > "${evidence_root}/build-products.sha256"
grep -E 'Athena_ENABLE_(MPI|OPENMP)|Kokkos_ENABLE_(CUDA|OPENMP|SERIAL|SYCL)|CMAKE_BUILD_TYPE' \
  "${build_root}/CMakeCache.txt" > "${evidence_root}/cache-contract.txt"

ctest_command=(ctest --test-dir "${build_root}" --output-on-failure -j 4)
printf '%q ' "${ctest_command[@]}" > "${evidence_root}/ctest-command.txt"
printf '\n' >> "${evidence_root}/ctest-command.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --exact \
  --kill-on-bad-exit=1 bash -c 'set -euo pipefail; export OMP_NUM_THREADS=8 OMP_PROC_BIND=false KOKKOS_NUM_THREADS=8; "$@"' \
  bash "${ctest_command[@]}" > "${evidence_root}/ctest.log" 2>&1
grep -F '100% tests passed' "${evidence_root}/ctest.log" >/dev/null

convergence=${source_root}/tst/unit/z4c/z4c_vertex_dynamic_linear_wave_test.py
input2=${source_root}/tst/inputs/z4c_vc_dynamic_linear_wave_2d_cartesian.athinput
input3=${source_root}/tst/inputs/z4c_vc_dynamic_linear_wave_3d_cartesian.athinput
for order in 2 4 6; do
  if test "${order}" -eq 6; then
    resolutions=(24 36 48)
  else
    resolutions=(16 32 64)
  fi
  for dimensions in 2 3; do
    input=${input2}
    test "${dimensions}" -eq 3 && input=${input3}
    label=o${order}-${dimensions}d
    command=(
      "${python_bin}" "${convergence}"
      --athena "${athena}"
      --input "${input}"
      --work-dir "${campaign_root}/runs/phase0-${label}"
      --dimensions "${dimensions}"
      --order "${order}"
      --integrator rk4
      --resolutions "${resolutions[@]}"
    )
    printf '%q ' "${command[@]}" > "${evidence_root}/${label}-command.txt"
    printf '\n' >> "${evidence_root}/${label}-command.txt"
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --exact \
      --kill-on-bad-exit=1 "${command[@]}" \
      > "${evidence_root}/${label}.log" 2>&1
    grep -F 'PASS: native-VC' "${evidence_root}/${label}.log" >/dev/null
  done
done

ctest --test-dir "${build_root}" -N > "${evidence_root}/ctest-inventory.txt"
grep -F 'athena.z4c_cc_selector_equivalence' "${evidence_root}/ctest-inventory.txt" >/dev/null
printf 'PHASE0_BASE_REPRODUCED\n' > "${evidence_root}/verdict.txt"

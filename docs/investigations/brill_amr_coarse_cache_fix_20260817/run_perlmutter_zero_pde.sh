#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-coarse-cache-ab651f0e-v5-20260817
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
run_root=${root}/run
evidence=${run_root}/evidence
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
iris_root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/source/iris
iris_archive=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/build/iris-host/src/libiris_athenak_interpolator.a
expected_commit=ab651f0ebd113f8718fefbf6d802976e6b3e8738
expected_tree=fae0f46e52717ab0e9a3f6c3ffc2dbbc0261b96f
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
expected_restart=83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea

test "${SLURM_JOB_NAME:-}" = cartoon-brill-cache-ab651-v5
test "${SLURM_JOB_NUM_NODES:-}" = 1
test "${SLURM_NTASKS:-}" = 2
test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_GPUS_PER_NODE:-}" = 1
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${root}/build"
test ! -e "${root}/run"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_commit}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -z "$(git -C "${source_root}/kokkos" status --short)"
test "$(sha256sum "${root}/input/brill_n256_pre_event_c1721.00014.rst" | awk '{print $1}')" = \
  "${expected_restart}"

mkdir -p "${evidence}"
finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run_root}/allocation.status"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${evidence}/kokkos-status.final"
  find "${run_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum > "${run_root}/SHA256SUMS"
  sha256sum "${run_root}/SHA256SUMS" > "${run_root}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export COLLAPSE_ROOT=${root}
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${evidence}/environment.txt"
module -t list > "${evidence}/modules.txt" 2>&1
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence}/hosts.txt"
node=$(head -1 "${evidence}/hosts.txt")
scontrol show node "${node}" -o > "${evidence}/node.txt"
grep -E 'ActiveFeatures=[^ ]*gpu' "${evidence}/node.txt" >/dev/null
grep -E 'ActiveFeatures=[^ ]*hbm40g' "${evidence}/node.txt" >/dev/null

configure=(cmake -S "${source_root}" -B "${build_root}"
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC
  -DPROBLEM=built_in_pgens -DBUILD_TESTING=ON -DAthena_BUILD_UNIT_TESTS=ON
  -DAthena_ENABLE_IRISK_INTERPOLATOR=ON -DIRISK_ROOT="${iris_root}"
  -DIRISK_INTERPOLATOR_LIBRARY="${iris_archive}"
  -DAthena_SINGLE_PRECISION=OFF -DAthena_ENABLE_MPI=ON
  -DAthena_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DKokkos_ENABLE_OPENMP=ON
  -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ARCH_AMPERE80=ON)
printf '%q ' "${configure[@]}" > "${evidence}/configure-command.txt"
printf '\n' >> "${evidence}/configure-command.txt"
/usr/bin/time -p -o "${evidence}/configure-time.txt" "${configure[@]}" \
  > "${evidence}/configure.log" 2> "${evidence}/configure.err"
printf 'configure|0\n' >> "${evidence}/phase-status.tsv"

build=(cmake --build "${build_root}" --parallel 8 --target athena
  athena_z4c_amr_jump_diagnostic_unit_test
  athena_z4c_chi_prolongation_unit_test
  athena_z4c_coarse_cache_ownership_unit_test)
printf '%q ' "${build[@]}" > "${evidence}/build-command.txt"
printf '\n' >> "${evidence}/build-command.txt"
/usr/bin/time -p -o "${evidence}/build-time.txt" "${build[@]}" \
  > "${evidence}/build.log" 2> "${evidence}/build.err"
printf 'build|0\n' >> "${evidence}/phase-status.tsv"

ctest --test-dir "${build_root}" --show-only=json-v1 \
  -R '^athena\.z4c_(coarse_cache_ownership|coarse_cache_ownership_mpi2|amr_jump_diagnostic|chi_prolongation|amr_chi_refresh_static|cartoon_amr_static)$' \
  > "${evidence}/focused-tests-inventory.json"
single_test_step=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=map_gpu:0 \
  --cpu-bind=cores --exact --kill-on-bad-exit=1)
"${single_test_step[@]}" \
  "${build_root}/athena_z4c_chi_prolongation_unit_test" \
  > "${evidence}/chi-prolongation-test.log" 2>&1
"${single_test_step[@]}" \
  "${build_root}/athena_z4c_coarse_cache_ownership_unit_test" \
  > "${evidence}/coarse-cache-ownership-test.log" 2>&1
ctest --test-dir "${build_root}" --output-on-failure \
  -R '^athena\.z4c_(coarse_cache_ownership_mpi2|amr_jump_diagnostic|amr_chi_refresh_static|cartoon_amr_static)$' \
  > "${evidence}/focused-tests.log" 2>&1
printf 'focused_tests|0\n' >> "${evidence}/phase-status.tsv"

case_root=${run_root}/zero_pde
mkdir -p "${case_root}/bindings"
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
executable=${build_root}/src/athena
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 --time=00:25:00 "${python_bin}" "${wrapper}"
  --evidence-dir "${case_root}/bindings" --require-cuda -- "${executable}"
  -r "${root}/input/brill_n256_pre_event_c1721.00014.rst"
  -i "${root}/input/continuation.athinput" -d "${case_root}"
  job/basename=brill_n256_cache_fix_zero_pde
  problem/brill_global_coefficients_file="${root}/input/brill_global_48x32.coefficients"
  problem/constraint_summary_file=brill_n256_cache_fix_zero_pde-constraints.dat
  z4c/amr_jump_output_basename=z4c_amr_jump_cache_fix
  z4c/amr_jump_post_cycles=0)
printf '%q ' "${command[@]}" > "${case_root}/command.txt"
printf '\n' >> "${case_root}/command.txt"
"${command[@]}" > "${case_root}/run.log" 2>&1
printf 'zero_pde|0\n' >> "${evidence}/phase-status.tsv"

event=$(find "${case_root}/z4c_amr_jump_cache_fix/rank0000" -maxdepth 1 \
  -type d -name 'event_c00001722_*' -print -quit)
test -n "${event}"
test -f "${event}/t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION/aggregate.json"
grep -F 'after T5 and before the next RHS' "${case_root}/run.log" >/dev/null
sha256sum "${executable}" "${build_root}/CMakeCache.txt" \
  "${root}/input/brill_n256_pre_event_c1721.00014.rst" \
  "${event}/t0_00_ACCEPTED_OLD_STATE/aggregate.json" \
  "${event}/t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION/aggregate.json" \
  > "${evidence}/qualification.sha256"
printf 'terminal|0\n' >> "${evidence}/phase-status.tsv"

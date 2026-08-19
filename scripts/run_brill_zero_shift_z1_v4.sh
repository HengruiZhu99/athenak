#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-shift-controls-sourcecompat-v4-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
run=${root}/run/arm-zero-shift
evidence=${root}/evidence/z1
input=${source_root}/docs/investigations/brill_zero_shift_advection_controls_20260818/arm_zero_shift.athinput
history=${root}/input/n128-authority-hierarchy.jsonl
coeff=${root}/input/brill_global_48x32.coefficients
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
iris_root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/source/iris
iris_archive=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/build/iris-host/src/libiris_athenak_interpolator.a
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_GPUS:-}" = 1
scontrol show job "${SLURM_JOB_ID}" -o | grep -E 'QOS=(gpu_)?shared_interactive' >/dev/null
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test "$(git -C "${source_root}" rev-parse HEAD)" = \
  "$(git -C "${source_root}" rev-parse '@{upstream}')"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = 6739bc623081648af9e752b616d9671527922cbf
test "$(sha256sum "${history}" | awk '{print $1}')" = d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555
test "$(sha256sum "${coeff}" | awk '{print $1}')" = ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
test ! -e "${build_root}" && test ! -e "${run}"
mkdir -p "${evidence}" "${run}/bindings"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${evidence}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
  find "${run}" "${evidence}" -type f ! -name terminal.sha256 -print0 |
    sort -z | xargs -0 sha256sum > "${evidence}/terminal.sha256"
  exit "${status}"
}
trap finish EXIT

export COLLAPSE_ROOT=${root}
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${evidence}/environment.txt"
git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}' > "${evidence}/source-identity.txt"
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${evidence}/node.txt"
nvidia-smi -L > "${evidence}/nvidia-smi-L.txt"

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

build=(cmake --build "${build_root}" --parallel 8 --target athena
  athena_amr_history_format_unit_test athena_z4c_chi_preupdate_diagnostic_unit_test)
printf '%q ' "${build[@]}" > "${evidence}/build-command.txt"
printf '\n' >> "${evidence}/build-command.txt"
/usr/bin/time -p -o "${evidence}/build-time.txt" "${build[@]}" \
  > "${evidence}/build.log" 2> "${evidence}/build.err"
ctest --test-dir "${build_root}" --output-on-failure \
  -R '^athena\.(amr_history_format|amr_history_extension_static|z4c_chi_preupdate_diagnostic|z4c_chi_preupdate_diagnostic_static|z4c_shift_control_static)$' \
  > "${evidence}/focused-tests.log" 2>&1
grep -F '100% tests passed, 0 tests failed out of 5' "${evidence}/focused-tests.log" >/dev/null
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" > "${evidence}/build-products.sha256"

command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=single:1
  "${python_bin}" "${wrapper}" --evidence-dir "${run}/bindings" --require-cuda --
  "${build_root}/src/athena" -i "${input}" -d "${run}"
  mesh_refinement/amr_history_file="${history}"
  problem/brill_global_coefficients_file="${coeff}")
printf '%q ' "${command[@]}" > "${run}/command.txt"
printf '\n' >> "${run}/command.txt"
status=0
(cd "${run}" && "${command[@]}") > "${run}/run.log" 2>&1 || status=$?
printf '%s\n' "${status}" > "${run}/exit-status.txt"
test "${status}" = 0 || test "${status}" = 1
test -s "${run}/arm_zero_shift.z4c.user.hst"
test -s "${run}/shift_invariant_check.csv"
awk -F, 'NR>1 && $4 != 0 {exit 1}' "${run}/shift_invariant_check.csv"
exit "${status}"

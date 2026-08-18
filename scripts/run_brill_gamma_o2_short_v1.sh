#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-shift-controls-90838417-v1-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
run=${root}/run/arm-gamma-o2-short
evidence=${root}/evidence/u2-short
restart=${root}/input/n256-replay-cycle4096.rst
history=${root}/input/n128-authority-hierarchy.jsonl
coeff=${root}/input/brill_global_48x32.coefficients
override=${source_root}/docs/investigations/brill_zero_shift_advection_controls_20260818/arm_gamma_o2_restart_override.athinput
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_GPUS:-}" = 1
scontrol show job "${SLURM_JOB_ID}" -o | grep -E 'QOS=(gpu_)?shared_interactive' >/dev/null
test "$(git -C "${source_root}" rev-parse HEAD)" = 908384176de0d5081304bfd91fd31d006f969049
test "$(sha256sum "${restart}" | awk '{print $1}')" = 2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9
test "$(sha256sum "${history}" | awk '{print $1}')" = d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555
test -x "${build_root}/src/athena" && test ! -e "${run}"
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
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${evidence}/node.txt"
nvidia-smi -L > "${evidence}/nvidia-smi-L.txt"
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" > "${evidence}/build-products.sha256"

command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=single:1
  "${python_bin}" "${wrapper}" --evidence-dir "${run}/bindings" --require-cuda --
  "${build_root}/src/athena" -r "${restart}" -i "${override}" -d "${run}"
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file="${history}"
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=arm-gamma-o2-short-constraints.dat)
printf '%q ' "${command[@]}" > "${run}/command.txt"
printf '\n' >> "${run}/command.txt"
status=0
(cd "${run}" && "${command[@]}") > "${run}/run.log" 2>&1 || status=$?
printf '%s\n' "${status}" > "${run}/exit-status.txt"
test "${status}" = 0 || test "${status}" = 1
test -s "${run}/arm_gamma_o2_short.z4c.user.hst"
exit "${status}"

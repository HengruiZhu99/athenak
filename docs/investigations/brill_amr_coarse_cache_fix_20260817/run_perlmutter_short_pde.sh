#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-coarse-cache-ab651f0e-v6-pde-20260817
source_root=${root}/source/athenak
executable=${root}/build/athena
run_root=${root}/run
evidence=${run_root}/evidence
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
expected_commit=ab651f0ebd113f8718fefbf6d802976e6b3e8738
expected_tree=fae0f46e52717ab0e9a3f6c3ffc2dbbc0261b96f
expected_executable=2c05dc123811c00c2cb6239e11d4f074bb85e605da467ecc6557f74becd9352f
expected_restart=83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea

test "${SLURM_JOB_NAME:-}" = cartoon-brill-cache-ab651-v6
test "${SLURM_JOB_NUM_NODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_GPUS_PER_NODE:-}" = 1
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${run_root}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_commit}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --short)"
test "$(sha256sum "${executable}" | awk '{print $1}')" = "${expected_executable}"
test "$(sha256sum "${root}/input/brill_n256_pre_event_c1721.00014.rst" | awk '{print $1}')" = "${expected_restart}"

mkdir -p "${evidence}"
finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run_root}/allocation.status"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
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

case_root=${run_root}/patched_short_pde
mkdir -p "${case_root}/bindings"
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 --time=01:30:00 "${python_bin}" "${wrapper}"
  --evidence-dir "${case_root}/bindings" --require-cuda -- "${executable}"
  -r "${root}/input/brill_n256_pre_event_c1721.00014.rst"
  -i "${root}/input/continuation.athinput" -d "${case_root}"
  job/basename=brill_n256_cache_fix_pde
  problem/brill_global_coefficients_file="${root}/input/brill_global_48x32.coefficients"
  problem/constraint_summary_file=brill_n256_cache_fix_pde-constraints.dat
  z4c/amr_jump_output_basename=z4c_amr_jump_cache_fix_pde)
printf '%q ' "${command[@]}" > "${case_root}/command.txt"
printf '\n' >> "${case_root}/command.txt"
"${command[@]}" > "${case_root}/run.log" 2>&1
printf 'patched_short_pde|0\n' >> "${evidence}/phase-status.tsv"
grep -F 'Terminating on time limit' "${case_root}/run.log" >/dev/null
grep -F 'time=1.250000e+01' "${case_root}/run.log" >/dev/null
sha256sum "${executable}" "${root}/input/brill_n256_pre_event_c1721.00014.rst" \
  "${case_root}/run.log" > "${evidence}/qualification.sha256"
printf 'terminal|0\n' >> "${evidence}/phase-status.tsv"

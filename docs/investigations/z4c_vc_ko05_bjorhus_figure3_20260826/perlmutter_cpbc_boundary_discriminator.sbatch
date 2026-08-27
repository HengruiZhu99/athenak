#!/usr/bin/env bash
#SBATCH --account=m3328_g
#SBATCH --qos=shared_interactive
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NERSC shared_interactive requires reserving 32 CPU cores per allocated GPU.
# Every srun step below remains explicitly capped at 16 CPU cores.
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=z4c-cpbc-abcd

set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-ko05-bjorhus-figure3-20260826
source_root=${campaign}/source-cpbc
build_root=${campaign}/build-cuda-mpi-cpbc
runner=${campaign}/bundle/perlmutter_cpbc_case.sh
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh

export COLLAPSE_ROOT=/pscratch/sd/h/hzhu/collapse-critical-perlmutter
# shellcheck source=/dev/null
source "${profile}"

test_log=${campaign}/evidence/cpbc_cuda_tests.log
mkdir -p "${campaign}/evidence"
{
  date --iso-8601=seconds
  sha256sum \
    "${build_root}/athena_z4c_sommerfeld_derivatives_unit_test" \
    "${build_root}/athena_z4c_full_constraint_bjorhus_unit_test" \
    "${build_root}/athena_z4c_shared_geometry_policy_unit_test"
  for test_binary in \
    athena_z4c_sommerfeld_derivatives_unit_test \
    athena_z4c_full_constraint_bjorhus_unit_test \
    athena_z4c_shared_geometry_policy_unit_test; do
    srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 \
      --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
      "${build_root}/${test_binary}"
  done
} 2>&1 | tee "${test_log}"

export CAMPAIGN_ROOT=${campaign}
export SOURCE_ROOT=${source_root}
export BUILD_ROOT=${build_root}
export RUN_TLIM=6.5
overall_status=0
run_case() {
  set +e
  "${runner}"
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then overall_status=1; fi
}

export DOMAIN=rout16
export BOUNDARY_RHS=sommerfeld
export HISTORY_MODE=record
export HISTORY_FILE=${campaign}/authority/cpbc_rout16_original.jsonl
export RUN_ROOT=${campaign}/runs/cpbc_A_rout16_original
run_case
chmod 0440 "${HISTORY_FILE}"

export BOUNDARY_RHS=full_constraint_bjorhus
export HISTORY_MODE=replay
export RUN_ROOT=${campaign}/runs/cpbc_B_rout16_bjorhus
run_case

export DOMAIN=rout128
export BOUNDARY_RHS=sommerfeld
export HISTORY_MODE=record
export HISTORY_FILE=${campaign}/authority/cpbc_rout128_original.jsonl
export RUN_ROOT=${campaign}/runs/cpbc_C_rout128_original
run_case
chmod 0440 "${HISTORY_FILE}"

export BOUNDARY_RHS=full_constraint_bjorhus
export HISTORY_MODE=replay
export RUN_ROOT=${campaign}/runs/cpbc_D_rout128_bjorhus
run_case

exit "${overall_status}"

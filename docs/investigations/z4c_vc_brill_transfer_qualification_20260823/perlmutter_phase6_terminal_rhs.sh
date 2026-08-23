#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
source_root=${campaign_root}/source/athenak
build_root=${campaign_root}/build/current-cuda-mpi
evidence_root=${campaign_root}/evidence/phase6-fixed-brill-terminal-rhs
base_run_root=${campaign_root}/runs/phase6-fixed-brill-retry1
run_root=${campaign_root}/runs/phase6-fixed-brill-terminal-rhs
expected_source=278b63a740a947de55ad8bdd1c333095c68fedcd
athena=${build_root}/src/athena

test -n "${SLURM_JOB_ID:-}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${athena}"
test ! -e "${evidence_root}"
test ! -e "${run_root}"
mkdir -p "${evidence_root}" "${run_root}"

finish() {
  status=$?
  set +e
  printf '%s\n' "${status}" > "${evidence_root}/exit-status.txt"
  find "${evidence_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 -r sha256sum > "${evidence_root}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
scontrol show job "${SLURM_JOB_ID}" > "${evidence_root}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
nvidia-smi -L > "${evidence_root}/nvidia-smi.txt"
sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
  > "${evidence_root}/build-products.sha256"

for resolution in 128 256 512; do
  case "${resolution}" in
    128) terminal_cycle=385; stride=1 ;;
    256) terminal_cycle=771; stride=2 ;;
    512) terminal_cycle=1541; stride=4 ;;
  esac
  restart=$(find "${base_run_root}/N${resolution}/rst" \
    -name "*.00001.rst" -print -quit)
  test -n "${restart}"
  run=${run_root}/N${resolution}
  mkdir -p "${run}"
  export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC=${run}/rhs
  export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC_STRIDE=${stride}
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 \
    "${athena}" -r "${restart}" -d "${run}" \
    time/nlim="$((terminal_cycle + 1))" time/tlim=6.0 \
    > "${evidence_root}/N${resolution}.stdout.log" \
    2> "${evidence_root}/N${resolution}.stderr.log"
  test -f "${run}/rhs.rank000000.csv"
  grep -F "cycle=$((terminal_cycle + 1))" \
    "${evidence_root}/N${resolution}.stdout.log" >/dev/null
done

find "${run_root}" -name 'rhs.rank*.csv' -print0 | sort -z | \
  xargs -0 -r sha256sum > "${evidence_root}/raw-rhs-before-compression.sha256"
find "${run_root}" -name 'rhs.rank*.csv' -print0 | sort -z | xargs -0 -r gzip -9
find "${run_root}" -type f -print0 | sort -z | xargs -0 -r sha256sum \
  > "${evidence_root}/run-products.sha256"
printf '%s\n' 'FIXED_GRID_BRILL_T5_ONE_STAGE_RHS_DIAGNOSTIC_CAPTURED' \
  > "${evidence_root}/verdict.txt"

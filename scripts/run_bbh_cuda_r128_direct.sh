#!/bin/bash
# Run one formulation directly on della-vis1's visible CUDA device.
# Each AthenaK invocation stops cleanly after 55 minutes and this driver
# resumes from the newest checkpoint until the physical t=100 limit is met.

set -eo pipefail

: "${FORMULATION:?run with FORMULATION=z4c or FORMULATION=pcgh}"
if [[ "${FORMULATION}" != z4c && "${FORMULATION}" != pcgh ]]; then
  echo "FORMULATION must be z4c or pcgh" >&2
  exit 2
fi

source /home/hz0693/athenak_env
set -u

repo=/home/hz0693/athenak-pcgh-cuda-20260904
athena="${repo}/build-cuda-a100/src/athena"
input="${repo}/inputs/z4c/twopuncture/bbh_headon_${FORMULATION}_cuda_r128_t100.athinput"
run_root=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128
run_dir="${run_root}/${FORMULATION}-r128-t100"
driver_log="${run_dir}/direct-driver.log"

mkdir -p "${run_dir}"
exec 9>"${run_dir}/.direct-run.lock"
if ! flock -n 9; then
  echo "Another direct driver already holds ${run_dir}/.direct-run.lock" >&2
  exit 3
fi

exec > >(tee -a "${driver_log}") 2>&1

echo "DIRECT_DRIVER_START=$(date --iso-8601=seconds)"
echo "RUN_DIR=${run_dir}"
echo "FORMULATION=${FORMULATION}"
echo "DRIVER_PID=$$"

if [[ ! -f "${run_dir}/used_input.athinput" ]]; then
  cp "${input}" "${run_dir}/used_input.athinput"
fi
sha256sum "${athena}" "${input}" > "${run_dir}/provenance.sha256"
git -C "${repo}" status --short > "${run_dir}/git-status.txt"
git -C "${repo}" rev-parse HEAD > "${run_dir}/git-commit.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

segment=0
while true; do
  segment=$((segment + 1))
  printf -v segment_id "%03d" "${segment}"
  segment_log="${run_dir}/direct-segment-${segment_id}.log"
  if [[ -e "${segment_log}" ]]; then
    printf -v segment_id "%03d-%s" "${segment}" "$(date +%s)"
    segment_log="${run_dir}/direct-segment-${segment_id}.log"
  fi

  run_args=(-i "${input}")
  shopt -s nullglob
  restarts=("${run_dir}"/rst/*.rst)
  shopt -u nullglob
  if ((${#restarts[@]})); then
    restart="${restarts[${#restarts[@]}-1]}"
    run_args=(-r "${restart}" -i "${input}")
    echo "SEGMENT=${segment_id} RESTART=${restart}"
  else
    echo "SEGMENT=${segment_id} FRESH_START=1"
  fi

  set +e
  (
    cd "${run_dir}"
    "${athena}" \
      "${run_args[@]}" \
      job/basename="bbh_${FORMULATION}_cuda_r128_t100" \
      -t 00:55:00
  ) 2>&1 | tee "${segment_log}"
  athena_status=${PIPESTATUS[0]}
  set -e

  if grep -q "Terminating on time limit" "${segment_log}"; then
    echo "DIRECT_DRIVER_COMPLETE=$(date --iso-8601=seconds)"
    exit 0
  fi
  if ((athena_status != 0)); then
    echo "AthenaK segment ${segment_id} failed with status ${athena_status}" >&2
    exit "${athena_status}"
  fi
  if ! grep -q "Terminating on wall clock limit" "${segment_log}"; then
    echo "AthenaK segment ${segment_id} ended without a recognized clean termination marker" >&2
    exit 4
  fi
  echo "SEGMENT=${segment_id} CLEAN_WALL_STOP=$(date --iso-8601=seconds)"
done

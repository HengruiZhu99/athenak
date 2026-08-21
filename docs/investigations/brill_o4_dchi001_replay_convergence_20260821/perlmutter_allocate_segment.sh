#!/usr/bin/env bash
set -euo pipefail

: "${COLLAPSE_ACCOUNT:=m3328_g}"
: "${JOB_NAME:?}"
: "${RUN_SCRIPT:?}"

exec salloc --account="${COLLAPSE_ACCOUNT}" --job-name="${JOB_NAME}" \
  --qos=shared_interactive --constraint='gpu&hbm80g' --nodes=1 \
  --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1 \
  --time=02:00:00 bash "${RUN_SCRIPT}"

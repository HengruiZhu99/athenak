#!/usr/bin/env bash
set -euo pipefail

: "${COLLAPSE_ACCOUNT:?set the NERSC account}"
: "${REMOTE_ROOT:?set the exact-final-source Perlmutter root}"

salloc --account="${COLLAPSE_ACCOUNT}" \
  --job-name=z4c-vc-final-cuda \
  --qos=shared_interactive --constraint='gpu&hbm80g' \
  --nodes=1 --ntasks=2 --cpus-per-task=16 --gpus-per-node=1 \
  --time=00:30:00 \
  env \
    CAMPAIGN_ROOT="${REMOTE_ROOT}/campaign-cuda-final-v1" \
    SOURCE_ROOT="${REMOTE_ROOT}/source" \
    BUILD_ROOT="${REMOTE_ROOT}/build" \
    bash "${REMOTE_ROOT}/input/run_perlmutter_final_source_qualification.sh"

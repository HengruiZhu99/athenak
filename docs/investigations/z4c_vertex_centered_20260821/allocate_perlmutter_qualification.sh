#!/usr/bin/env bash
set -euo pipefail

: "${COLLAPSE_ACCOUNT:?set the NERSC account}"
: "${REMOTE_ROOT:?}"

salloc --account="${COLLAPSE_ACCOUNT}" --job-name=z4c-vc-axis-fix-v2 \
  --qos=shared_interactive --constraint='gpu&hbm80g' --nodes=1 \
  --ntasks=2 --cpus-per-task=16 --gpus-per-node=1 --time=01:00:00 \
  env \
    CAMPAIGN_ROOT="${REMOTE_ROOT}/campaign-v2" \
    SOURCE_ROOT="${REMOTE_ROOT}/source" \
    BUILD_ROOT="${REMOTE_ROOT}/build" \
    FIXED_INPUT="${REMOTE_ROOT}/input/brill_vc_fixed_grid.athinput" \
    REPLAY_INPUT="${REMOTE_ROOT}/input/brill_vc_common_tree.athinput" \
    COEFFICIENT_FILE="${REMOTE_ROOT}/input/brill_global_48x32.coefficients" \
    AUTHORITY_FILE="${REMOTE_ROOT}/input/n256_amr_history.jsonl" \
    bash "${REMOTE_ROOT}/input/run_perlmutter_qualification.sh"

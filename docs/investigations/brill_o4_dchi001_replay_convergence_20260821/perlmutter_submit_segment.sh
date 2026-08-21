#!/usr/bin/env bash
set -euo pipefail

: "${COLLAPSE_ACCOUNT:=m3328_g}"
: "${JOB_NAME:?}"
: "${RUN_SCRIPT:?}"
: "${CAMPAIGN_ROOT:?}"

test -d "${CAMPAIGN_ROOT}"
test -x "${RUN_SCRIPT}"

# Retain the shared_interactive resource contract but submit it detached from
# the login connection. A queued salloc can otherwise be revoked when its
# client connection times out before resources become available.
exec sbatch --parsable --account="${COLLAPSE_ACCOUNT}" \
  --job-name="${JOB_NAME}" --qos=shared_interactive \
  --constraint='gpu&hbm80g' --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=32 --gpus-per-node=1 --time=02:00:00 \
  --output="${CAMPAIGN_ROOT}/allocation-%j.stdout" \
  --error="${CAMPAIGN_ROOT}/allocation-%j.stderr" \
  --export=ALL "${RUN_SCRIPT}"

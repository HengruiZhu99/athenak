#!/usr/bin/env bash
# Batch and interactive entry point. Submit the absolute path to this script.
set -eo pipefail
export PHOTON_CAMPAIGN="$1"
shift
cd "$PHOTON_CAMPAIGN"
if [[ -f site.sh ]]; then source ./site.sh "$@"; fi
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "${PHOTON_PYTHON:-python3}" -u "$1" "${@:2}"

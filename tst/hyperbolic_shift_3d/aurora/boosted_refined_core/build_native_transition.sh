#!/bin/bash -l
# Explicit build only; this script never submits an allocation or an evolution.
set -euo pipefail
root=${1:?Pass the existing Aurora home task root}
cd "$root"
module load oneapi/release/2025.3.1
exec /usr/bin/python3.11 launcher-boosted-refined-core-v1/native_transition_prepare.py build --root "$PWD"

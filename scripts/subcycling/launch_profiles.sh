#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
cd "$root"
# Run only after successful, provenance-bound build and comparison input preparation.
test -s executable.sha256
sha256sum -c executable.sha256
salloc --account=m3328_g --qos=shared_interactive --constraint='gpu&hbm80g' --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=01:30:00 --job-name=vc-subcycle-profile bash "$root/run_profiles.sh"

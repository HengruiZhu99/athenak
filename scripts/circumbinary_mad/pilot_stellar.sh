#!/bin/bash
# Submit from the campaign directory with eight GPU ranks on one node.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
sep=${1:-25}
out="$PWD/pilot-$SLURM_JOB_ID"
mkdir "$out"
mpirun --bind-to none --map-by slot -np "$SLURM_NTASKS" ./map.sh "$PWD/build/src/athena" \
    -i "$PWD/inputs/mad$sep-pilot.athinput" -d "$out" > "$out/log" 2>&1
module load anaconda3/2025.12
python3 "${MAD_SCRIPTS:?path to campaign scripts}/check_pilot.py" "$out" --separation "$sep" --input "$PWD/inputs/mad$sep-pilot.athinput" > "$out/qualification.json"
if [[ -f "$out/bin/mad$sep.midplane.00000.bin" ]]; then
    python3 "$MAD_SCRIPTS/disk_quality.py" "$out/bin/mad$sep.midplane.00000.bin" \
        "$out/bin/mad$sep.meridional.00000.bin" > "$out/initial-disk-quality.json"
fi

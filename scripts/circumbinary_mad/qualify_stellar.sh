#!/bin/bash
# Run the final analysis and release the two other cases after the 25M comparison.
set -eo pipefail
scripts=${1:?campaign scripts directory}
sep=${2:?separation}
module purge
module load anaconda3/2025.12
python3 "$scripts/late_flux.py" "$PWD" "$sep" > "seq$sep/late-flux.json.tmp"
mv "seq$sep/late-flux.json.tmp" "seq$sep/late-flux.json"
python3 "$scripts/late_flux.py" --check-report "seq$sep/late-flux.json"
if [[ $sep != 25 ]]; then exit 0; fi
for next in 15 35; do
    # Reserve the slot first to prevent duplicated science jobs on a manual replay.
    ledger="dispatch-$next"
    if [[ -s "$ledger/continue.job" ]]; then continue; fi
    mkdir "$ledger"
    flags=(-N1 --ntasks-per-node=8 -c4 --mem=450G --gres=gpu:8 -p gpu -A fpretori)
    pilot=$(sbatch --parsable "${flags[@]}" --qos=gpu-test -t 00:20:00 \
        --export=ALL,MAD_SCRIPTS="$scripts" -J "mad$next-pilot" "$scripts/pilot_stellar.sh" "$next")
    echo "$pilot" > "$ledger/pilot.job"
    first=$(sbatch --parsable "${flags[@]}" --dependency="afterok:$pilot" --qos=gpu-medium \
        -t 12:00:00 --export=ALL,ATHENA_WALL_LIMIT=11:48:00 -J "mad$next-disk" \
        "$scripts/run_stellar.sh" "$next")
    echo "$first" > "$ledger/first.job"
    sbatch --parsable "${flags[@]}" --dependency="afterok:$first" --qos=gpu-medium \
        -t 12:00:00 --export=ALL,ATHENA_WALL_LIMIT=11:48:00 -J "mad$next-continue" \
        "$scripts/segment.sh" "$next" "$first" 1 > "$ledger/continue.job"
done

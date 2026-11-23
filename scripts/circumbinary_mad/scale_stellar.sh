#!/bin/bash
# Short strong-scaling measurement after the initial one-node grid qualification.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
root="$PWD/scaling-$SLURM_JOB_ID"
mkdir "$root"
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
for nodes in 1 2; do
    ranks=$((8 * nodes))
    allocation="${hosts[0]}:8"
    if [[ $nodes == 2 ]]; then allocation="$allocation,${hosts[1]}:8"; fi
    for phase in finest outer; do
        work="$root/$nodes-$phase"
        mkdir "$work"
        overrides=()
        if [[ $phase == outer ]]; then
            overrides=(amr_criterion1/trun_fac_0=0.06 amr_criterion1/trun_fac_1=0.06)
        fi
        mpirun --host "$allocation" --bind-to none --map-by slot -np "$ranks" ./map.sh \
            "$PWD/build/src/athena" -i "$PWD/inputs/mad25-scale.athinput" -d "$work" \
            "${overrides[@]}" > "$work/log" 2>&1
        python3 ./ct_guard.py "$work/mad25.user.hst"
    done
done

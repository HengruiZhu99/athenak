#!/bin/bash
# Submit from a campaign directory containing build/src/athena and generated inputs/.
# One MPI rank per allocated GPU. Give Athena time to finish its wall-time checkpoint.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
sep=${1:?separation in total-mass units}
shift
out="${ATHENA_RUN_DIR:-$PWD/seq$sep}"
mkdir -p "$out"
exe="$PWD/build/src/athena"
restart=$(find "$out/rst" -maxdepth 1 -name "mad$sep.*.rst" 2>/dev/null | sort | tail -1 || true)
if [[ -n "$restart" ]]; then
    start=(-r "$restart")
else
    start=(-i "$PWD/inputs/mad$sep-run.athinput")
fi
log="$out/slurm-$SLURM_JOB_ID.log"
mpirun --bind-to none --map-by slot -np "$SLURM_NTASKS" ./map.sh "$exe" "${start[@]}" \
    -d "$out" -t "${ATHENA_WALL_LIMIT:-03:52:00}" "$@" > "$log" 2>&1 &
simulation=$!
trap 'kill -TERM "$simulation" 2>/dev/null || true' TERM INT
history="$out/mad$sep.user.hst"
while kill -0 "$simulation" 2>/dev/null; do
    sleep 30 &
    wait $! || true
    if [[ -s "$history" ]] && ! python3 ./ct_guard.py "$history" "$out/.ct-offset"; then
        echo "Magnetic-field guard stopped job $SLURM_JOB_ID. Inspect $history and $log." >&2
        kill -TERM "$simulation" 2>/dev/null || true
        wait "$simulation" || true
        exit 2
    fi
done
wait "$simulation"
python3 ./ct_guard.py "$history" "$out/.ct-offset"

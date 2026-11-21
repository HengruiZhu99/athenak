#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=eq-select
# Submit after the eight-GPU factorial. A late square-domain comparison qualifies
# the cheapest candidate on each physical endpoint before any production choice.
set -eo pipefail
module purge
module load anaconda3/2025.12
set -u
suite=tst/unit/boltzmann/equator
base=$SLURM_SUBMIT_DIR/..
sample=$base/measure-$1
python "$suite/performance.py" "$sample"/{square-full,square-half,coarse-full,coarse-half} > "$sample/cost.jsonl"
winner=$(python - "$sample/cost.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
matched = {(r["GPUs"], r["host"], tuple(r["pci_devices"]), r["initial_data_sha256"],
            r["executable_sha256"], r["start_time"], r["end_time"]) for r in rows}
if len(matched) != 1 or rows[0]["GPUs"] != 8:
    raise ValueError("unmatched layouts in the timing sample")
print(min(rows, key=lambda r:r["GPU_hours"])["layout"])
PY
)
echo "Eight-GPU timing winner $winner"
if [[ $winner == square-full ]]; then
    echo "The optimized layouts did not reduce the measured cost."
    exit 0
fi
candidate=(--square)
[[ $winner != coarse-* ]] || candidate+=(--coarse)
[[ $winner != *-half ]] || candidate+=(--half)
for family in j09508 j10600_refined; do
    end=74
    ref_limit=08:00:00
    trial_limit=06:00:00
    if [[ $family == j10600_refined ]]; then
        end=150.68796599118204
        ref_limit=12:00:00
        trial_limit=08:00:00
    fi
    ref=$(sbatch --parsable -t "$ref_limit" -J "eq-square-$family" "$suite/stellar.sh" \
        "$family-square-full" "$family" "$end" --square)
    run=$(sbatch --parsable -t "$trial_limit" -J "eq-best-$family" "$suite/stellar.sh" \
        "$family-$winner" "$family" "$end" "${candidate[@]}")
    review=$(sbatch --parsable --dependency="afterany:$ref:$run" "$suite/review.sh" \
        "$family" "$ref" "$run" "$family-square-full" "$family-$winner" "square-$family")
    echo "$family reference=$ref trial=$run review=$review"
    if [[ $family == j10600_refined ]]; then
        gate=$(sbatch --parsable --dependency="afterok:$review" "$suite/outward_continuation.sh" \
            "$ref" "$run" "$winner")
        echo "$family conditional 60 M continuation=$gate"
    fi
done

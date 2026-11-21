#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --job-name=eq-outward-gate
# Continue the two square layouts across the old boundary failure if their
# 30 M comparisons both recover an outward centroid and no verified trapping.
set -eo pipefail
module purge
module load anaconda3/2025.12
set -u
suite=tst/unit/boltzmann/equator
base=$SLURM_SUBMIT_DIR/..
review=$base/review-square-j10600_refined
python - "$review" <<'PY'
import csv,json,sys
from pathlib import Path
p=Path(sys.argv[1])
v=json.loads((p/"summary.json").read_text())
last=list(csv.DictReader((p/"timed-profiles.csv").open()))[-1]
trial=float(last["mean_Rcirc_M"])
reference=trial-float(last["delta_mean_Rcirc_M"])
if float(last["t_M"])<30-1e-7 or min(reference,trial)<=2.5 or any("first_verified_in_window" in r for r in v.values()):
    raise SystemExit("30 M outward conditions not met; retain the endpoint evidence for review")
PY
family=j10600_refined
end=301.3759319823641
ref=$(sbatch --parsable -t 10:00:00 -J eq-outward60-ref --export="ALL,EQUATOR_PARENT=$base/measure-$1/$family-square-full" \
        "$suite/stellar.sh" "$family-square-full" "$family" "$end")
trial=$(sbatch --parsable -t 10:00:00 -J eq-outward60-best --export="ALL,EQUATOR_PARENT=$base/measure-$2/$family-$3" \
        "$suite/stellar.sh" "$family-$3" "$family" "$end")
post=$(sbatch --parsable --dependency="afterany:$ref:$trial" "$suite/review.sh" "$family" "$ref" "$trial" \
       "$family-square-full" "$family-$3" square-j10600-to60)
echo "60 M comparison reference=$ref trial=$trial review=$post"

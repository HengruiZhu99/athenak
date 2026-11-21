#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --job-name=eq-review
set -eo pipefail
module purge
module load anaconda3/2025.12
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
family=$1
full=$2
half=$3
reference_tag=${4:-$family-full}
trial_tag=${5:-$family-half}
result_tag=${6:-$family}
base=$SLURM_SUBMIT_DIR/..
data=${EQUATOR_DATA:-/scratch/gpfs/FPRETORI/hzhu/photon-boundary-20261120/compact_spin_survey}
seed=${EQUATOR_WITNESS_SEED:-$base/j09508-trapped.json}
if [[ ! -s "$seed" ]]; then
    seed=$(mktemp)
    trap 'rm -f "$seed"' EXIT
    tar -xJOf reports/equatorial-reflection/horizon-replay-inputs.tar.xz j09508/seed.json > "$seed"
fi
python tst/unit/boltzmann/equator/endpoint_review.py "$base/measure-$full/$reference_tag" \
    "$base/measure-$half/$trial_tag" "$data/$family/ring.adm.json" "$base/review-$result_tag" --seed "$seed"

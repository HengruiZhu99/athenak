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
base=$SLURM_SUBMIT_DIR/..
data=${EQUATOR_DATA:-/scratch/gpfs/FPRETORI/hzhu/photon-boundary-20261120/compact_spin_survey}
opts=()
if [[ "$family" = j09508 ]]; then
    opts=(--seed "$base/j09508-trapped.json")
fi
python tst/unit/boltzmann/equator/endpoint_review.py "$base/measure-$full/$family-full" \
    "$base/measure-$half/$family-half" "$data/$family/ring.adm.json" "$base/review-$family" "${opts[@]}"

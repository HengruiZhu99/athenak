#!/usr/bin/env bash
#SBATCH --account=m3328_g
#SBATCH --qos=shared_interactive
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=z4c-shock-n256
#SBATCH --output=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/slurm/n256-seg000-%j.out
#SBATCH --error=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/slurm/n256-seg000-%j.err

set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828
qualification_job=${QUALIFICATION_JOB:?set passed CUDA qualification job ID}
test -f "${campaign}/evidence/cuda-qualification-${qualification_job}/disposition"
grep -Fxq PERLMUTTER_CUDA_QUALIFICATION_PASS \
  "${campaign}/evidence/cuda-qualification-${qualification_job}/disposition"

export RUN_ROOT=${campaign}/runs/n256_reference_shock_seg000
export HISTORY_FILE=${campaign}/authority/n256_reference_shock_authority.jsonl
export RUN_TLIM=30.0
exec "${campaign}/perlmutter_run_reference_n256.sh"

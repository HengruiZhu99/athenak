#!/bin/bash
#SBATCH --job-name=pcgh_R16_smr128
#SBATCH --qos=gpu-test
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-r16-smr128-20260905-1640/slurm-%A_%a.log
set -e
source /home/hz0693/athenak_env
set -u
campaign=/scratch/gpfs/FPRETORI/hz0693/pcgh-r16-smr128-20260905-1640
source_root=/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/monitor-source
spacings=(8 10 12)
n=${spacings[$SLURM_ARRAY_TASK_ID]}
cd "$source_root"
# Retain the exact executable used for the completed R16/P1 screening dynamics.
expected=96bc34157896c164904a6ff426e84f4c97d95a7b8c8c1a43971940fc744e60b1
actual=$(sha256sum build-hybrid-monitor-cuda/src/athena)
test "${actual%% *}" = "$expected"
python3 analysis/pc_gh_regular_extension/run_hybrid_stage.py \
  --inputs "$campaign/inputs" --build build-hybrid-monitor-cuda \
  --output "$campaign/runs/h$n" --stage smr --pattern "R16-smr128-h$n-t20.athinput"

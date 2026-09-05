#!/bin/bash
#SBATCH --job-name=pcgh_hybrid_core_hybrids
#SBATCH --qos=gpu-test
#SBATCH --array=0-2%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/core-hybrids-%A_%a.log
set -e
source /home/hz0693/athenak_env
set -u
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/monitor-source
names=(R4 R16 P1)
candidate=${names[$SLURM_ARRAY_TASK_ID]}
python3 analysis/pc_gh_regular_extension/run_hybrid_stage.py \
  --inputs ../inputs-final --build build-hybrid-monitor-cuda \
  --output "../core-hybrids/$candidate" --stage screen --pattern "core256-$candidate.athinput"

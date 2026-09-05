#!/bin/bash
#SBATCH --job-name=pcgh_hybrid_R4_resume
#SBATCH --qos=gpu-test
#SBATCH --dependency=afterany:13481379_0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/core-hybrids-R4-resume-%j.log
set -e
source /home/hz0693/athenak_env
set -u
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/monitor-source
python3 analysis/pc_gh_regular_extension/resume_hybrid_screen.py \
  --build build-hybrid-monitor-cuda --run ../core-hybrids/R4/core256-R4

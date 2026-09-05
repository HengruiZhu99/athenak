#!/bin/bash
#SBATCH --job-name=pcgh_hybrid_pulses
#SBATCH --qos=gpu-test
#SBATCH --array=0-5%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/pulses-%A_%a.log
set -e
source /home/hz0693/athenak_env
set -u
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/source
names=(C1 C4 C16 R4 R16 P1)
candidate=${names[$SLURM_ARRAY_TASK_ID]}
python3 analysis/pc_gh_regular_extension/run_hybrid_stage.py \
  --inputs ../inputs-qualified-controls --build build-hybrid-cuda \
  --output "../pulses/$candidate" --stage pulses --pattern "$candidate-*.athinput"
python3 analysis/pc_gh_regular_extension/verify_hybrid_pulses.py \
  ../pulses/"$candidate"/"$candidate"-*/ > "../pulses/$candidate/verification.log" 2>&1

#!/bin/bash
#SBATCH --job-name=pcgh_hybrid_reproduce
#SBATCH --qos=gpu-test
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/reproduction-%A_%a.log
set -e
source /home/hz0693/athenak_env
set -u
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/source
case "$SLURM_ARRAY_TASK_ID" in
  0) case_name=uniform-C1-original ;;
  1) case_name=core256-C16-original ;;
  *) exit 2 ;;
esac
python3 analysis/pc_gh_regular_extension/run_hybrid_stage.py \
  --inputs ../inputs-validated --build build-hybrid-cuda \
  --output "../reproduction/$case_name" --stage reproduction \
  --pattern "$case_name.athinput"

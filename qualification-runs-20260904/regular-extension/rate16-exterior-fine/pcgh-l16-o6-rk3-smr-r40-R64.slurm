#!/bin/bash
#SBATCH --job-name=pcgh_o6_fine40
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/rate16-fine40-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 analysis/pc_gh_regular_extension/cuda_driver.py run --build build-regular-cuda --input ../inputs/rate16-exterior-fine/pcgh-l16-o6-rk3-smr-r40-R64.athinput --output ../runs/rate16-qualification/pcgh-l16-o6-rk3-smr-r40-R64 --wall-segment 00:55:00 --max-segments 1

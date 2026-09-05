#!/bin/bash
#SBATCH --job-name=pcgh-l16-o6-rk3-uniform-
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/rate16-pcgh-l16-o6-rk3-uniform-r24-R8-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 analysis/pc_gh_regular_extension/cuda_driver.py run --build build-regular-cuda --input ../inputs/rate16-qualification/pcgh-l16-o6-rk3-uniform-r24-R8.athinput --output ../runs/rate16-qualification/pcgh-l16-o6-rk3-uniform-r24-R8

#!/bin/bash
#SBATCH --job-name=pcgh_o2_bbh_l0
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/binary-order2-l0-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 analysis/pc_gh_regular_extension/cuda_driver.py run --build build-binary-cuda --input ../inputs/binary-order2/headon-o2-rk4-l0-k1-t100.athinput --output ../runs/binary-order2/headon-o2-rk4-l0-k1-t100 --wall-segment 00:55:00 --max-segments 1

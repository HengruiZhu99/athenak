#!/bin/bash
#SBATCH --job-name=pcgh_regular_gate
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=00:20:00

# Submit with: sbatch --chdir=FRESH_SOURCE_ROOT --output=UNIQUE_LOG this_script \
#   BUILD_DIRECTORY INPUT_FILE UNIQUE_RUN_DIRECTORY
# Resource syntax follows the repository's previously used Della smoke job.
# Verify its continued availability using sinfo/sacctmgr before submitting.
set -eo pipefail
source /home/hz0693/athenak_env
set -u
python3 analysis/pc_gh_regular_extension/cuda_driver.py run \
  --build "$1" --input "$2" --output "$3" --wall-segment 00:15:00

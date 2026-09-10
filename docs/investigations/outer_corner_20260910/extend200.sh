#!/bin/bash
#SBATCH --account=m3328_g
#SBATCH --qos=shared_interactive
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --job-name=n256-profile
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
cd /pscratch/sd/h/hzhu/corner-fix-tests-20260910
scontrol show job "$SLURM_JOB_ID" > slurm-extend-job.txt
/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3 -u extend200.py

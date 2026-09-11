#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
cd "$root"
scontrol show job "$SLURM_JOB_ID" > profiles-slurm.txt
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1
/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3 -u "$root/run_profiles.py"

#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=30G
#SBATCH --time=00:08:00
#SBATCH --job-name=eq-AMR-test
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx anaconda3/2025.12
set -u
export EQUATOR_SRUN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export KOKKOS_DEVICE_ID=0 OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,cuda_copy,cuda_ipc
python -u tst/unit/boltzmann/equatorial_amr.py build/src/athena "../gpu-amr-$SLURM_JOB_ID"

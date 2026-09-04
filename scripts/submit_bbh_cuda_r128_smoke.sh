#!/bin/bash
#SBATCH --job-name=bbh_r128_smoke
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/logs/smoke_%j.out
#SBATCH --error=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/logs/smoke_%j.err

set -euo pipefail

source /home/hz0693/athenak_env

repo=/home/hz0693/athenak-pcgh-cuda-20260904
athena="${repo}/build-cuda-mpi-a100/src/athena"
run_root=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128

mkdir -p "${run_root}/logs"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

for formulation in pcgh z4c; do
  input="${repo}/inputs/z4c/twopuncture/bbh_headon_${formulation}_cuda_r128_t100.athinput"
  run_dir="${run_root}/smoke-${formulation}-${SLURM_JOB_ID}"
  mkdir -p "${run_dir}"
  cp "${input}" "${run_dir}/used_input.athinput"
  sha256sum "${athena}" "${input}" > "${run_dir}/provenance.sha256"
  srun --nodes=1 \
    --ntasks="${SLURM_NTASKS:-4}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    "${athena}" \
    -i "${input}" \
    -d "${run_dir}" \
    time/nlim=1 \
    time/tlim=0.1 \
    job/basename="${formulation}_gpu_smoke" 2>&1 | tee "${run_dir}/run.log"
done

date

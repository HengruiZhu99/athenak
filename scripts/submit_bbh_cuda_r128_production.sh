#!/bin/bash
# Submit with FORMULATION=z4c or FORMULATION=pcgh.
#SBATCH --job-name=bbh_r128_t100
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/logs/production_%x_%j.out
#SBATCH --error=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/logs/production_%x_%j.err

set -euo pipefail

: "${FORMULATION:?submit with --export=ALL,FORMULATION=z4c or pcgh}"
if [[ "${FORMULATION}" != z4c && "${FORMULATION}" != pcgh ]]; then
  echo "FORMULATION must be z4c or pcgh" >&2
  exit 2
fi

source /home/hz0693/athenak_env

repo=/home/hz0693/athenak-pcgh-cuda-20260904
athena="${repo}/build-cuda-mpi-a100/src/athena"
input="${repo}/inputs/z4c/twopuncture/bbh_headon_${FORMULATION}_cuda_r128_t100.athinput"
run_root=/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128
run_dir="${run_root}/${FORMULATION}-r128-t100"
segment_log="${run_dir}/segment-${SLURM_JOB_ID}.log"

mkdir -p "${run_root}/logs" "${run_dir}"
if [[ ! -f "${run_dir}/used_input.athinput" ]]; then
  cp "${input}" "${run_dir}/used_input.athinput"
fi
sha256sum "${athena}" "${input}" > "${run_dir}/provenance-${SLURM_JOB_ID}.sha256"
git -C "${repo}" status --short > "${run_dir}/git-status-${SLURM_JOB_ID}.txt"
git -C "${repo}" rev-parse HEAD > "${run_dir}/git-commit-${SLURM_JOB_ID}.txt"

run_args=(-i "${input}")
if compgen -G "${run_dir}/rst/*.rst" >/dev/null; then
  restarts=("${run_dir}"/rst/*.rst)
  restart="${restarts[${#restarts[@]}-1]}"
  run_args=(-r "${restart}" -i "${input}")
  echo "RESTART=${restart}"
fi

date
echo "RUN_DIR=${run_dir}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

srun --nodes=1 \
  --ntasks="${SLURM_NTASKS:-4}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${athena}" \
  "${run_args[@]}" \
  -d "${run_dir}" \
  job/basename="bbh_${FORMULATION}_cuda_r128_t100" \
  -t 00:55:00 2>&1 | tee "${segment_log}"

date
if grep -q "Terminating on wall clock limit" "${segment_log}"; then
  echo "Submitting the next clean-restart segment."
  sbatch --export=ALL,FORMULATION="${FORMULATION}" \
    "${repo}/scripts/submit_bbh_cuda_r128_production.sh"
elif grep -q "Terminating on time limit" "${segment_log}"; then
  echo "Reached tlim; no further segment submitted."
else
  echo "Run ended without a recognized clean termination marker." >&2
  exit 3
fi

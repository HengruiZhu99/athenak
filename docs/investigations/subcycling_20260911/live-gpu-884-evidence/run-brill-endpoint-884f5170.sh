#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
out="$root/brill-endpoint-884f5170"
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1
[ "$(cat "$root/live_gpu_884f5170/exit-status")" = 0 ]
trap 'echo $? > "$out/job-exit-status"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 nvidia-smi > "$out/gpu.txt"
python3 "$root/endpoint-tools-e206f34f/run_brill_endpoint.py" "$out/manifest.json" > "$out/runner.log" 2>&1

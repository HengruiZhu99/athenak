#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
[ "$(cat "$root/build-live-884f5170/status")" = 0 ]
[ "$(git -C "$root/source" rev-parse --short=8 HEAD)" = 38c1fab4 ]
(cd "$root/source"; sha256sum -c "$root/vc-live-884f5170.files.sha256")
out="$root/live_gpu_884f5170"
mkdir "$out"
trap 'echo $? > "$out/exit-status"' EXIT
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
git -C "$root/source" rev-parse HEAD > "$out/source-sha.txt"
sha256sum "$root/build/src/athena" > "$out/athena.sha256"
cp "$root/vc-live-884f5170.patch" "$root/vc-live-884f5170.files.sha256" "$out/"
# Each Python test launches several MPI programs. Give each a separate Slurm
# step; the hash in test JSON identifies this wrapper, actual binary hash above.
cat > "$out/athena-step" <<'WRAPPER'
#!/bin/bash
set -euo pipefail
exec srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 /pscratch/sd/h/hzhu/vc-subcycling-20260911/build/src/athena "$@"
WRAPPER
chmod +x "$out/athena-step"
python3 "$root/source/scripts/subcycling/test_live_subcycling.py" "$out/athena-step" "$out/fixed" --production-orders --production-gauge > "$out/fixed.log" 2>&1
python3 "$root/source/scripts/subcycling/test_live_amr.py" "$out/athena-step" "$out/amr" --production-orders > "$out/amr.log" 2>&1
python3 "$root/source/scripts/subcycling/test_live_amr.py" "$out/athena-step" "$out/mixed" --production-orders --mixed > "$out/mixed.log" 2>&1
python3 "$root/source/scripts/subcycling/test_checkpoint_probe.py" "$out/athena-step" "$out/temporal" --production-orders --production-gauge --dt .004 --fixed-duration .016 > "$out/temporal.log" 2>&1
sha256sum -c "$out/athena.sha256" > "$out/hash-verification.txt"

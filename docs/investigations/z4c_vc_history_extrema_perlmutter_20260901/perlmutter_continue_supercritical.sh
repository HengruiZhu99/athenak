#!/usr/bin/env bash
set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901
source_root=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/supercritical-horizon-20260830/source
python=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
athena=${campaign}/build/athena.history_extrema
run=${campaign}/runs/supercritical_Aminus0p050
restart=${run}/rst/n512_Aminus0p050_history_extrema.00001.rst

module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = \
  c3ef1c8b371eb3a447108d3b0acc115b34cfeaeb71337fda277e18d409a5b8c0
test -f "${restart}"
mkdir -p "${run}/rank-bindings-continuation"
exec > >(tee -a "${campaign}/allocation-supercritical-continuation.log") 2>&1

date -Is
hostname
scontrol show job "${SLURM_JOB_ID}"
cd "${run}"
set +e
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
  --kill-on-bad-exit=1 \
  "${python}" "${wrapper}" --evidence-dir "${run}/rank-bindings-continuation" \
  --require-cuda -- "${athena}" -r "${restart}" -t 00:25:00 \
  mesh_refinement/amr_history_mode=record \
  mesh_refinement/amr_history_file="${run}/n512_Aminus0p050_history_extrema.amr_history.jsonl" \
  time/tlim=38.652331986867424 time/nlim=-1 \
  output2/dt=1000 output3/dt=1000 output4/dt=1000 output5/dt=1000 \
  job/basename=n512_Aminus0p050_history_extrema \
  problem/constraint_summary_file=n512_Aminus0p050_history_extrema-constraints.dat \
  > stdout-continuation.log 2> stderr-continuation.log
status=$?
set -e
printf '%s\n' "${status}" > run-status-continuation
echo SUPERCRITICAL_CONTINUATION_STATUS=${status}
date -Is
exit 0

#!/usr/bin/env bash
set -euo pipefail

# Continue one record-mode run inside the same class of one-GPU interactive
# allocation.  This is used for KO Stage B/C and the optional late boundary
# control.  It never submits a batch job.
: "${PREVIOUS_RUN_ROOT:?completed record segment}"
: "${RUN_ROOT:?fresh continuation directory}"
: "${RUN_TLIM:?new coordinate-time target}"
: "${AUTHORITY_NAME:?authority filename from the previous segment}"

source_root=${SOURCE_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-derefine-slot-repair-20260824/source}
build_root=${BUILD_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-native-authority-20260824/build-cuda-mpi-deterministic-history}
profile=${PERLMUTTER_PROFILE:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh}
python_bin=${NERSC_PYTHON:-/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3}
expected_source=d63519328214a6315a9cc1f7d5e4a1aa4bca21b0
expected_tree=9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test ! -e "${RUN_ROOT}"
test -s "${PREVIOUS_RUN_ROOT}/${AUTHORITY_NAME}"
restart_file=$(find "${PREVIOUS_RUN_ROOT}/rst" -maxdepth 1 -type f -name '*.rst' |
  LC_ALL=C sort | tail -n 1)
test -s "${restart_file}"

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
cp "${PREVIOUS_RUN_ROOT}/${AUTHORITY_NAME}" "${RUN_ROOT}/"
cp "${PREVIOUS_RUN_ROOT}/brill_global_128x32.coefficients" "${RUN_ROOT}/"
cp "${PREVIOUS_RUN_ROOT}"/*.athinput "${RUN_ROOT}/"

export COLLAPSE_ROOT=${COLLAPSE_ROOT:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter}
# shellcheck source=/dev/null
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta
athena=${build_root}/src/athena
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
scontrol show job "${SLURM_JOB_ID}" > "${RUN_ROOT}/evidence/slurm-job.txt"
env | LC_ALL=C sort > "${RUN_ROOT}/evidence/environment.txt"
{
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  sha256sum "${athena}" "${restart_file}" "${RUN_ROOT}/${AUTHORITY_NAME}"
  module list
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${RUN_ROOT}/evidence/provenance.txt" 2>&1

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits --loop=1 > "${RUN_ROOT}/evidence/gpu-memory.csv" 2>&1 &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${RUN_ROOT}/rank-bindings" --require-cuda -- "${athena}"
  -r "${restart_file}" -t 00:20:00
  mesh_refinement/amr_history_mode=record
  mesh_refinement/amr_history_file="${AUTHORITY_NAME}"
  time/tlim="${RUN_TLIM}" time/nlim=-1)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/command.txt"
printf '\n' >> "${RUN_ROOT}/command.txt"
set +e
(cd "${RUN_ROOT}" && "${command[@]}" > stdout.log 2> stderr.log)
status=$?
set -e
cleanup_monitor
trap - EXIT
printf '%d\n' "${status}" > "${RUN_ROOT}/athena-exit"
printf '%s\n' "${restart_file}" > "${RUN_ROOT}/evidence/restart-input.txt"
awk -F, 'BEGIN{max=-1} /^[0-9]/{gsub(/ /,"",$5); if (($5+0)>max) max=$5+0} END{print max}' \
  "${RUN_ROOT}/evidence/gpu-memory.csv" > "${RUN_ROOT}/evidence/peak-hbm-used-mib.txt"
find "${RUN_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${RUN_ROOT}/SHA256SUMS"
(cd "${RUN_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
exit "${status}"

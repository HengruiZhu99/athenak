#!/usr/bin/env bash
set -euo pipefail

# Run this script only inside an already granted one-GPU interactive allocation.
# It never submits a batch job.
: "${EXPERIMENT:?boundary or ko}"
: "${CASE_LABEL:?n128, n256, or n512}"
: "${AMR_HISTORY_MODE:?record or replay}"
: "${RUN_ROOT:?fresh absolute run directory}"
: "${RUN_TLIM:?coordinate-time target}"
: "${CAMPAIGN_BUNDLE:?directory containing the two campaign inputs}"

source_root=${SOURCE_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-derefine-slot-repair-20260824/source}
build_root=${BUILD_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-native-authority-20260824/build-cuda-mpi-deterministic-history}
coefficient_source=${COEFFICIENT_SOURCE:-${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/evidence/brill_global_128x32_origin_control.coefficients}
profile=${PERLMUTTER_PROFILE:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh}
python_bin=${NERSC_PYTHON:-/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3}
expected_source=d63519328214a6315a9cc1f7d5e4a1aa4bca21b0
expected_tree=9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_GPUS_ON_NODE:-1}" -ge 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test -x "${build_root}/src/athena"
test -r "${profile}"
test -r "${coefficient_source}"
test ! -e "${RUN_ROOT}"

case "${CASE_LABEL}" in
  n128) root_nx1=64;  root_nx2=128; mb_nx1=16; mb_nx2=16 ;;
  n256) root_nx1=128; root_nx2=256; mb_nx1=32; mb_nx2=32 ;;
  n512) root_nx1=256; root_nx2=512; mb_nx1=64; mb_nx2=64 ;;
  *) printf 'unsupported CASE_LABEL=%s\n' "${CASE_LABEL}" >&2; exit 2 ;;
esac

case "${EXPERIMENT}" in
  boundary)
    input_name=brill_vc_rout128_smr.athinput
    diss=0.02
    ;;
  ko)
    test "${CASE_LABEL}" = n256
    test "${AMR_HISTORY_MODE}" = record
    : "${DISS:?KO experiment requires DISS}"
    case "${DISS}" in
      0.02|0.05|0.10|0.20|0.50) ;;
      *) printf 'unsupported KO DISS=%s\n' "${DISS}" >&2; exit 2 ;;
    esac
    input_name=brill_vc_rout16_ko.athinput
    diss=${DISS}
    ;;
  *) printf 'unsupported EXPERIMENT=%s\n' "${EXPERIMENT}" >&2; exit 2 ;;
esac

case "${AMR_HISTORY_MODE}" in
  record)
    test "${CASE_LABEL}" = n256
    history_name=${AUTHORITY_NAME:-n256_authority.jsonl}
    ;;
  replay)
    : "${AUTHORITY_FILE:?replay requires AUTHORITY_FILE}"
    test -s "${AUTHORITY_FILE}"
    history_name=n256_authority.jsonl
    ;;
  *) printf 'unsupported AMR_HISTORY_MODE=%s\n' "${AMR_HISTORY_MODE}" >&2; exit 2 ;;
esac

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
cp "${CAMPAIGN_BUNDLE}/${input_name}" "${RUN_ROOT}/"
cp "${coefficient_source}" "${RUN_ROOT}/brill_global_128x32.coefficients"
if test "${AMR_HISTORY_MODE}" = replay; then
  cp "${AUTHORITY_FILE}" "${RUN_ROOT}/${history_name}"
fi

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
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
    "${RUN_ROOT}/${input_name}" "${RUN_ROOT}/brill_global_128x32.coefficients"
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
  -i "${input_name}" -t 00:20:00
  mesh/nx1="${root_nx1}" mesh/nx2="${root_nx2}"
  meshblock/nx1="${mb_nx1}" meshblock/nx2="${mb_nx2}"
  mesh_refinement/amr_history_mode="${AMR_HISTORY_MODE}"
  mesh_refinement/amr_history_file="${history_name}"
  time/tlim="${RUN_TLIM}" time/nlim=-1 z4c/diss="${diss}"
  job/basename="${CASE_LABEL}_${EXPERIMENT}_${AMR_HISTORY_MODE}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${CASE_LABEL}_${EXPERIMENT}_${AMR_HISTORY_MODE}-constraints.dat")

printf '%q ' "${command[@]}" > "${RUN_ROOT}/command.txt"
printf '\n' >> "${RUN_ROOT}/command.txt"
set +e
(cd "${RUN_ROOT}" && "${command[@]}" > stdout.log 2> stderr.log)
status=$?
set -e
cleanup_monitor
trap - EXIT
printf '%d\n' "${status}" > "${RUN_ROOT}/athena-exit"
awk -F, 'BEGIN{max=-1} /^[0-9]/{gsub(/ /,"",$5); if (($5+0)>max) max=$5+0} END{print max}' \
  "${RUN_ROOT}/evidence/gpu-memory.csv" > "${RUN_ROOT}/evidence/peak-hbm-used-mib.txt"
find "${RUN_ROOT}" -type f ! -name SHA256SUMS -print0 | \
  LC_ALL=C sort -z | xargs -0r sha256sum > "${RUN_ROOT}/SHA256SUMS"
(cd "${RUN_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
exit "${status}"

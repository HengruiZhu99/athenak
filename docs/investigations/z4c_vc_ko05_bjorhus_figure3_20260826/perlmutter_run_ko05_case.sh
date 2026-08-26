#!/usr/bin/env bash
set -euo pipefail

# Run inside one already-granted Perlmutter A100 allocation. Each invocation
# owns a fresh directory; restart continuations are separate immutable segments.
: "${CASE_LABEL:?n128, n256, or n512}"
: "${HISTORY_MODE:?record or replay}"
: "${HISTORY_FILE:?absolute shared authority path}"
: "${RUN_ROOT:?fresh absolute segment directory}"
: "${RUN_TLIM:?coordinate-time target}"
: "${CAMPAIGN_BUNDLE:?directory containing input}"

source_root=${SOURCE_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-derefine-slot-repair-20260824/source}
build_root=${BUILD_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-native-authority-20260824/build-cuda-mpi-deterministic-history}
coefficient_source=${COEFFICIENT_SOURCE:-${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/evidence/brill_global_128x32_origin_control.coefficients}
profile=${PERLMUTTER_PROFILE:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh}
python_bin=${NERSC_PYTHON:-/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3}
expected_source=${EXPECTED_SOURCE:-d63519328214a6315a9cc1f7d5e4a1aa4bca21b0}
expected_tree=${EXPECTED_TREE:-9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4}
expected_exe=${EXPECTED_EXE:-3a395bfdaf217d617fee43d2cbcd38e7a13c2a0f4207e3a764c3513eb8c0405f}
expected_coeff=${EXPECTED_COEFF:-1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10}
expected_cache=${EXPECTED_CMAKE_CACHE:-}
irisk_library=${IRISK_LIBRARY:-}
expected_irisk=${EXPECTED_IRISK_LIBRARY_SHA256:-}

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_GPUS_ON_NODE:-1}" -ge 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
athena=${build_root}/src/athena
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${coefficient_source}" | awk '{print $1}')" = "${expected_coeff}"
if [[ -n "${expected_cache}" ]]; then
  test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"
fi
if [[ -n "${expected_irisk}" ]]; then
  test -s "${irisk_library}"
  test "$(sha256sum "${irisk_library}" | awk '{print $1}')" = "${expected_irisk}"
fi
test -r "${profile}"
test ! -e "${RUN_ROOT}"

case "${CASE_LABEL}" in
  n128) root_nx1=64;  root_nx2=128; mb_nx1=16; mb_nx2=16; max_nmb=4096 ;;
  n256) root_nx1=128; root_nx2=256; mb_nx1=32; mb_nx2=32; max_nmb=4096 ;;
  n512) root_nx1=256; root_nx2=512; mb_nx1=64; mb_nx2=64; max_nmb=2048 ;;
  *) printf 'unsupported CASE_LABEL=%s\n' "${CASE_LABEL}" >&2; exit 2 ;;
esac
case "${CASE_LABEL}:${HISTORY_MODE}" in
  n256:record) ;;
  n128:replay|n256:replay|n512:replay) test -s "${HISTORY_FILE}" ;;
  *) printf 'unsupported case/history contract\n' >&2; exit 2 ;;
esac

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
cp "${CAMPAIGN_BUNDLE}/brill_vc_rout128_ko05.athinput" "${RUN_ROOT}/"
cp "${coefficient_source}" "${RUN_ROOT}/brill_global_128x32.coefficients"

export COLLAPSE_ROOT=${COLLAPSE_ROOT:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter}
# shellcheck source=/dev/null
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta

root=$(cd "${RUN_ROOT}" && pwd)
scontrol show job "${SLURM_JOB_ID}" > "${root}/evidence/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${root}/evidence/node.txt"
env | LC_ALL=C sort > "${root}/evidence/environment.txt"
{
  date --iso-8601=seconds
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
    "${root}/brill_vc_rout128_ko05.athinput" \
    "${root}/brill_global_128x32.coefficients"
  if [[ -n "${expected_irisk}" ]]; then sha256sum "${irisk_library}"; fi
  module list
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${root}/evidence/provenance.txt" 2>&1

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits --loop=1 > "${root}/evidence/gpu-memory.csv" 2>&1 &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
finish() {
  code=$?
  trap - EXIT
  set +e
  cleanup_monitor
  printf '%s\n' "${code}" > "${root}/orchestration-status"
  git -C "${source_root}" status --porcelain=v1 > "${root}/evidence/source-status.final"
  awk -F, 'BEGIN{max=-1} /^[0-9]/{gsub(/ /,"",$5); if (($5+0)>max) max=$5+0} END{print max}' \
    "${root}/evidence/gpu-memory.csv" > "${root}/evidence/peak-hbm-used-mib.txt"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 | \
    LC_ALL=C sort -z | xargs -0r sha256sum > "${root}/SHA256SUMS"
  (cd "${root}" && sha256sum -c SHA256SUMS >/dev/null)
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${code}"
}
trap finish EXIT

restart=(-i brill_vc_rout128_ko05.athinput)
if [[ -n "${RESTART_FILE:-}" ]]; then
  test -s "${RESTART_FILE}"
  restart=(-r "${RESTART_FILE}")
  sha256sum "${RESTART_FILE}" > "${root}/evidence/restart-input.sha256"
fi

wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${root}/rank-bindings" --require-cuda -- "${athena}"
  "${restart[@]}" -t 01:40:00
  mesh/nx1="${root_nx1}" mesh/nx2="${root_nx2}"
  meshblock/nx1="${mb_nx1}" meshblock/nx2="${mb_nx2}"
  mesh_refinement/max_nmb_per_rank="${max_nmb}"
  mesh_refinement/amr_history_mode="${HISTORY_MODE}"
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1 z4c/diss=0.50
  job/basename="${CASE_LABEL}_ko05_${HISTORY_MODE}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${CASE_LABEL}_ko05_${HISTORY_MODE}-constraints.dat")
printf '%q ' "${command[@]}" > "${root}/command.txt"; printf '\n' >> "${root}/command.txt"

set +e
(cd "${root}" && "${command[@]}" > stdout.log 2> stderr.log)
status=$?
set -e
printf 'athena_exit=%s\n' "${status}" > "${root}/run-status"
if [[ ${status} -ne 0 ]]; then
  disposition=FAIL_CLOSED_NUMERICAL_OR_RUNTIME_FAILURE
elif grep -Fq 'Terminating on time limit' "${root}/stdout.log"; then
  disposition=REACHED_TLIM
elif grep -Fq 'Terminating on wall clock limit' "${root}/stdout.log"; then
  disposition=HEALTHY_WALLTIME_RESTART_REQUIRED
else
  disposition=UNCLASSIFIED_CLEAN_EXIT
fi
printf '%s\n' "${disposition}" > "${root}/disposition"
if [[ "${disposition}" = UNCLASSIFIED_CLEAN_EXIT ]]; then exit 2; fi
exit "${status}"

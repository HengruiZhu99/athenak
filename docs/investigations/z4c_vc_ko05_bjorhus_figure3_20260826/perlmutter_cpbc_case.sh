#!/usr/bin/env bash
set -euo pipefail

# One fresh N256 A100 segment for the bounded boundary discriminator.
: "${DOMAIN:?rout16 or rout128}"
: "${BOUNDARY_RHS:?sommerfeld or full_constraint_bjorhus}"
: "${HISTORY_MODE:?record or replay}"
: "${HISTORY_FILE:?absolute authority path}"
: "${RUN_ROOT:?fresh absolute run directory}"
: "${RUN_TLIM:?coordinate-time target}"

campaign=${CAMPAIGN_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-ko05-bjorhus-figure3-20260826}
source_root=${SOURCE_ROOT:-${campaign}/source-cpbc}
build_root=${BUILD_ROOT:-${campaign}/build-cuda-mpi-cpbc}
profile=${PERLMUTTER_PROFILE:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh}
python_bin=${NERSC_PYTHON:-/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3}
coefficient_source=${COEFFICIENT_SOURCE:-${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/evidence/brill_global_128x32_origin_control.coefficients}
expected_source=${EXPECTED_SOURCE:-d39822c6522688749fe5ead8025907bc055f02f8}
expected_tree=${EXPECTED_TREE:-6054251064451beb5f435726aa9489e1a7979a96}
expected_exe=${EXPECTED_EXE:-235aea2e0cb306dc2894d3436e0338f2e9f6e09bdf1e322e29ed296df3afcb79}
expected_cache=${EXPECTED_CACHE:-9a07287d8d28b864215426bb7ada004846b1c6c7ee44d1e18fee1e95f3e7c802}
expected_coeff=${EXPECTED_COEFF:-1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10}

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_GPUS_ON_NODE:-1}" -ge 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test "$(sha256sum "${build_root}/src/athena" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"
test "$(sha256sum "${coefficient_source}" | awk '{print $1}')" = "${expected_coeff}"
test ! -e "${RUN_ROOT}"

case "${DOMAIN}" in
  rout16)
    input_source=${source_root}/docs/investigations/z4c_vc_boundary_convergence_ko_20260824/brill_vc_rout16_ko.athinput
    ;;
  rout128)
    input_source=${source_root}/docs/investigations/z4c_vc_ko05_bjorhus_figure3_20260826/brill_vc_rout128_ko05.athinput
    ;;
  *) printf 'unsupported DOMAIN=%s\n' "${DOMAIN}" >&2; exit 2 ;;
esac
case "${BOUNDARY_RHS}" in
  sommerfeld|full_constraint_bjorhus) ;;
  *) printf 'unsupported BOUNDARY_RHS=%s\n' "${BOUNDARY_RHS}" >&2; exit 2 ;;
esac
case "${HISTORY_MODE}" in
  record) test ! -e "${HISTORY_FILE}" ;;
  replay) test -s "${HISTORY_FILE}" ;;
  *) printf 'unsupported HISTORY_MODE=%s\n' "${HISTORY_MODE}" >&2; exit 2 ;;
esac

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
input_name=$(basename "${input_source}")
cp "${input_source}" "${RUN_ROOT}/${input_name}"
cp "${coefficient_source}" "${RUN_ROOT}/brill_global_128x32.coefficients"

export COLLAPSE_ROOT=${COLLAPSE_ROOT:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter}
# shellcheck source=/dev/null
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta

root=$(cd "${RUN_ROOT}" && pwd)
athena=${build_root}/src/athena
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
scontrol show job "${SLURM_JOB_ID}" > "${root}/evidence/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${root}/evidence/node.txt"
env | LC_ALL=C sort > "${root}/evidence/environment.txt"
{
  date --iso-8601=seconds
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
    "${root}/${input_name}" "${root}/brill_global_128x32.coefficients"
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
  awk -F, 'BEGIN{max=-1} /^[0-9]/{gsub(/ /,"",$5); if (($5+0)>max) max=$5+0} END{print max}' \
    "${root}/evidence/gpu-memory.csv" > "${root}/evidence/peak-hbm-used-mib.txt"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 | \
    LC_ALL=C sort -z | xargs -0r sha256sum > "${root}/SHA256SUMS"
  (cd "${root}" && sha256sum -c SHA256SUMS >/dev/null)
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${code}"
}
trap finish EXIT

basename_out=n256_${DOMAIN}_${BOUNDARY_RHS}
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${root}/rank-bindings" --require-cuda -- "${athena}"
  -i "${input_name}" -t 00:25:00
  mesh/nx1=128 mesh/nx2=256 meshblock/nx1=32 meshblock/nx2=32
  mesh_refinement/max_nmb_per_rank=4096
  mesh_refinement/amr_history_mode="${HISTORY_MODE}"
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1 z4c/diss=0.50
  z4c/boundary_rhs="${BOUNDARY_RHS}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${basename_out}-constraints.dat"
  job/basename="${basename_out}")
printf '%q ' "${command[@]}" > "${root}/command.txt"
printf '\n' >> "${root}/command.txt"

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

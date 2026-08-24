#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?}"
: "${BUILD_ROOT:?}"
: "${INPUT_FILE:?}"
: "${COEFFICIENT_FILE:?}"
: "${CAMPAIGN_ROOT:?}"
: "${SEGMENT_ROOT:?fresh path required}"
: "${CASE_LABEL:?n128, n256, or n512}"
: "${HISTORY_MODE:?record or replay}"
: "${HISTORY_FILE:?}"
: "${ROOT_NX1:?}"
: "${ROOT_NX2:?}"
: "${MB_NX1:?}"
: "${MB_NX2:?}"
: "${MAX_NMB_PER_RANK:?}"
: "${RUN_TLIM:?}"

expected_source=6ad9cf4048af6a93aa73cf9940fc78c3b439c8fe
expected_exe=87b86be33725ddb0d55dbd3484fdb36cf570f3436e7677c0e6bbcae823773204
expected_coeff=1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10

test ! -e "${SEGMENT_ROOT}"
case "${SLURM_JOB_QOS:-}" in
  shared_interactive|gpu_shared_interactive|gpu_shared_interactive_ss11) ;;
  *) echo "unexpected qos ${SLURM_JOB_QOS:-missing}" >&2; exit 2 ;;
esac
test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_GPUS_PER_NODE:-}" = 1
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${expected_source}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)"

athena="${BUILD_ROOT}/src/athena"
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${COEFFICIENT_FILE}" | awk '{print $1}')" = "${expected_coeff}"
case "${CASE_LABEL}:${HISTORY_MODE}:${ROOT_NX1}:${ROOT_NX2}:${MB_NX1}:${MB_NX2}:${MAX_NMB_PER_RANK}" in
  n128:replay:64:128:16:16:4096) ;;
  n256:record:128:256:32:32:4096) ;;
  n512:replay:256:512:64:64:2048) ;;
  *) echo "unsupported production case contract" >&2; exit 2 ;;
esac
if [[ "${HISTORY_MODE}" = replay ]]; then test -s "${HISTORY_FILE}"; fi

profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
wrapper="${SOURCE_ROOT}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py"
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
export COLLAPSE_ROOT="${CAMPAIGN_ROOT}"
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU

mkdir -p "${SEGMENT_ROOT}/rank-bindings" "${SEGMENT_ROOT}/evidence"
root=$(cd "${SEGMENT_ROOT}" && pwd)
cp "${INPUT_FILE}" "${root}/brill_vc_figure3.athinput"
cp "${COEFFICIENT_FILE}" "${root}/brill_global_128x32.coefficients"

finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s\n' "${code}" > "${root}/orchestration-status"
  git -C "${SOURCE_ROOT}" status --porcelain=v1 > "${root}/evidence/source-status.final"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 |
    LC_ALL=C sort -z | xargs -0r sha256sum > "${root}/SHA256SUMS"
  (cd "${root}" && sha256sum -c SHA256SUMS >/dev/null)
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${code}"
}
trap finish EXIT

env | LC_ALL=C sort > "${root}/evidence/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${root}/evidence/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${root}/evidence/node.txt"
{
  date --iso-8601=seconds
  git -C "${SOURCE_ROOT}" rev-parse HEAD HEAD^{tree}
  git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${BUILD_ROOT}/CMakeCache.txt" \
    "${root}/brill_vc_figure3.athinput" \
    "${root}/brill_global_128x32.coefficients"
  module list
} > "${root}/evidence/provenance.txt" 2>&1

restart=(-i brill_vc_figure3.athinput)
if [[ -n "${RESTART_FILE:-}" ]]; then
  test -s "${RESTART_FILE}"
  restart=(-r "${RESTART_FILE}")
  sha256sum "${RESTART_FILE}" > "${root}/evidence/restart-input.sha256"
fi

command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${root}/rank-bindings" --require-cuda -- "${athena}"
  "${restart[@]}" -t 01:35:00
  mesh/nx1="${ROOT_NX1}" mesh/nx2="${ROOT_NX2}"
  meshblock/nx1="${MB_NX1}" meshblock/nx2="${MB_NX2}"
  mesh_refinement/max_nmb_per_rank="${MAX_NMB_PER_RANK}"
  mesh_refinement/amr_history_mode="${HISTORY_MODE}"
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1
  job/basename="${CASE_LABEL}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${CASE_LABEL}-constraints.dat")
printf '%q ' "${command[@]}" > "${root}/command.txt"; printf '\n' >> "${root}/command.txt"

set +e
(cd "${root}" && "${command[@]}" > stdout.log 2> stderr.log)
status=$?
set -e
printf 'athena_exit=%s\n' "${status}" > "${root}/run-status"
"${python_bin}" - "${root}/rank-bindings" <<'PY'
import json, pathlib, sys
files = sorted(pathlib.Path(sys.argv[1]).glob("rank_binding_*.json"))
assert len(files) == 1, files
record = json.loads(files[0].read_text())
assert record["rank"] == 0 and record["local_rank"] == 0
assert record["binding_verified"] is True
assert "NVIDIA A100-SXM4" in record["gpu_name"]
PY

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
if [[ -e "${HISTORY_FILE}" ]]; then sha256sum "${HISTORY_FILE}" > "${root}/evidence/history-after.sha256"; fi
if [[ "${disposition}" = UNCLASSIFIED_CLEAN_EXIT ]]; then exit 2; fi
exit "${status}"


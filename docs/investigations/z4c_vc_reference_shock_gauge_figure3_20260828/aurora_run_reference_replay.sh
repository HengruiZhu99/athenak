#!/usr/bin/env bash
set -euo pipefail

: "${CASE_LABEL:?n512 is required}"
: "${RUN_ROOT:?fresh absolute segment directory}"
: "${HISTORY_FILE:?absolute N256 authority}"
: "${RUN_TLIM:?coordinate-time ceiling}"
: "${QUALIFICATION_ID:?passed Aurora PVC qualification job ID}"
: "${RUN_NODES:=2}"
: "${RANKS_PER_NODE:=12}"
: "${ATHENA_WALL_LIMIT:=00:20:00}"

campaign=${CAMPAIGN_ROOT:-/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-reference-shock-gauge-figure3-aurora-20260828}
source_root=${SOURCE_ROOT:-${campaign}/source}
build_root=${BUILD_ROOT:-${campaign}/build-sycl-mpi-f8303c6b}
coefficient_source=${COEFFICIENT_SOURCE:-${campaign}/brill_global_128x32.coefficients}
workflow=${AURORA_WORKFLOW:-/lus/flare/projects/CompactBinaryMerger/hzhu/collapse-critical/collapse-critical-workflow}
irisk_root=${IRISK_ROOT_OVERRIDE:-/lus/flare/projects/CompactBinaryMerger/hzhu/collapse-critical/bbhk}
irisk_library=${irisk_root}/build/aurora-pvc/src/libiris_athenak_interpolator.a
tile_wrapper=/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh

expected_source=f8303c6be7eb214fa1e91b646123ee0d434b3698
expected_tree=7a585ca487b12351b084eb425bb812775849b001
expected_exe=aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13
expected_cache=8da40bcb47564d9184119ca207f9847a33a3d1b5bd2930627d705cda8fb36386
expected_input=6c694cf871a3d694d745f0fb58b279b6cd07516463ac8ad54f1c91d2689c90ba
expected_coeff=1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10
expected_irisk=380a90d5b1d9762fe7f9076edcb27fb4a209f4cd8c070da376c36284a438c7a1
expected_history=7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde

case "${CASE_LABEL}" in
  n128) root_nx1=64; root_nx2=128; mb_nx1=16; mb_nx2=16; max_nmb=2048 ;;
  n512) root_nx1=256; root_nx2=512; mb_nx1=64; mb_nx2=64; max_nmb=2048 ;;
  *) printf 'unsupported CASE_LABEL=%s (expected n128 or n512)\n' "${CASE_LABEL}" >&2; exit 2 ;;
esac

qualification=${campaign}/evidence/pvc-qualification-${QUALIFICATION_ID}
grep -Fxq AURORA_PVC_QUALIFICATION_PASS "${qualification}/disposition"
test -r "${PBS_NODEFILE:-}"
test "$(LC_ALL=C sort -u "${PBS_NODEFILE}" | wc -l)" -eq "${RUN_NODES}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
athena=${build_root}/src/athena
input=${source_root}/docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/brill_vc_reference_shock_gauge.athinput
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"
test "$(sha256sum "${input}" | awk '{print $1}')" = "${expected_input}"
test "$(sha256sum "${coefficient_source}" | awk '{print $1}')" = "${expected_coeff}"
test "$(sha256sum "${irisk_library}" | awk '{print $1}')" = "${expected_irisk}"
test "$(sha256sum "${HISTORY_FILE}" | awk '{print $1}')" = "${expected_history}"
test ! -e "${RUN_ROOT}"
if [[ -n "${RESTART_FILE:-}" ]]; then test -s "${RESTART_FILE}"; fi

mkdir -p "${RUN_ROOT}/evidence"
cp "${input}" "${RUN_ROOT}/"
cp "${coefficient_source}" "${RUN_ROOT}/brill_global_128x32.coefficients"
root=$(cd "${RUN_ROOT}" && pwd)

export ATHENAK_ROOT=${source_root}
export ATHENAK_BUILD_DIR=${build_root}
export IRISK_ROOT=${irisk_root}
# shellcheck source=/dev/null
source "${workflow}/profiles/aurora-pvc.sh"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export MPIR_CVAR_ENABLE_GPU=1 MPICH_GPU_SUPPORT_ENABLED=1

finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s\n' "${code}" > "${root}/orchestration-status"
  git -C "${source_root}" status --porcelain=v1 > "${root}/evidence/source-status.final"
  qstat -f "${PBS_JOBID}" > "${root}/evidence/qstat-final.txt" 2>&1
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
    -print0 | LC_ALL=C sort -z | xargs -0r sha256sum > "${root}/SHA256SUMS"
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${code}"
}
trap finish EXIT

qstat -f "${PBS_JOBID}" > "${root}/evidence/qstat-start.txt"
env | LC_ALL=C sort > "${root}/evidence/environment.txt"
{
  date --iso-8601=seconds
  hostname
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
    "${root}/brill_vc_reference_shock_gauge.athinput" \
    "${root}/brill_global_128x32.coefficients" "${irisk_library}" \
    "${HISTORY_FILE}"
  sed -n '1,11p' "${root}/brill_global_128x32.coefficients"
  module list
  sycl-ls
} > "${root}/evidence/provenance.txt" 2>&1

restart=(-i brill_vc_reference_shock_gauge.athinput)
if [[ -n "${RESTART_FILE:-}" ]]; then
  restart=(-r "${RESTART_FILE}")
  sha256sum "${RESTART_FILE}" > "${root}/evidence/restart-input.sha256"
fi

basename=${CASE_LABEL}_reference_shock_replay
total_ranks=$((RUN_NODES * RANKS_PER_NODE))
command=(mpiexec -n "${total_ranks}" -ppn "${RANKS_PER_NODE}" --depth 8
  --cpu-bind depth "${tile_wrapper}"
  "${athena}" "${restart[@]}" -t "${ATHENA_WALL_LIMIT}"
  mesh/nx1="${root_nx1}" mesh/nx2="${root_nx2}"
  meshblock/nx1="${mb_nx1}" meshblock/nx2="${mb_nx2}"
  mesh_refinement/max_nmb_per_rank="${max_nmb}"
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1
  z4c/lapse_shock_avoiding=true z4c/lapse_shock_avoiding_kappa=1
  z4c/telegraph_lapse=false z4c/shift_mode=prescribed_zero
  z4c/shift_invariant_diagnostic=true z4c/diss=0.50
  z4c/damp_kappa1=0 z4c/damp_kappa2=0
  job/basename="${basename}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${basename}-constraints.dat")
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

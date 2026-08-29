#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?fresh absolute run directory}"
: "${RUN_NODES:?number of allocated nodes}"
: "${RANKS_PER_NODE:?one rank per requested PVC tile}"
: "${RUN_TLIM:?coordinate-time ceiling}"
: "${RESTART_FILE:?absolute N512 restart}"
: "${HISTORY_FILE:?absolute AMR-history authority}"
: "${ATHENA_WALL_LIMIT:=00:08:00}"
: "${RUN_SMOKE:=0}"
: "${ALLOCATED_NODES:=${RUN_NODES}}"

campaign=/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-performance-aurora-20260829
reference=/lus/flare/projects/CompactBinaryMerger/hzhu/z4c-vc-reference-shock-gauge-figure3-aurora-20260828
source_root=${campaign}/source
build_root=${campaign}/build-sycl-mpi-9c98bd14
athena=${build_root}/src/athena
input=${source_root}/docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/brill_vc_reference_shock_gauge.athinput
coefficient_source=${reference}/brill_global_128x32.coefficients
workflow=/lus/flare/projects/CompactBinaryMerger/hzhu/collapse-critical/collapse-critical-workflow
irisk_root=/lus/flare/projects/CompactBinaryMerger/hzhu/collapse-critical/bbhk
irisk_library=${irisk_root}/build/aurora-pvc/src/libiris_athenak_interpolator.a
tile_wrapper=/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh

expected_source=62993e7bac8fbaed13f592834282ca09142a5c2d
expected_tree=339b8f6a134a50fe7916013fd96f5cf93ea3a58d
expected_exe=b070bf3b856be712134b0e38028304bbb2fde506aa271350f98b3d8ee243c1e2
expected_cache=cfb772dc21b7161565bac7e3ffeac1047a3ed9ccdde146eea4054a66f573d656
expected_input=d86595adeaf7d4c72e1f150d044cd5277f7b4e3507eea4e6d03f427c8dc38838
expected_coeff=1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10
expected_irisk=380a90d5b1d9762fe7f9076edcb27fb4a209f4cd8c070da376c36284a438c7a1
expected_history=7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde
expected_restart=44b8e55957d3b455adf24862d36946e08fc10465df7a30cc5f247ac0e19fa997

test -r "${PBS_NODEFILE:-}"
test "$(LC_ALL=C sort -u "${PBS_NODEFILE}" | wc -l)" -eq "${ALLOCATED_NODES}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"
test "$(sha256sum "${input}" | awk '{print $1}')" = "${expected_input}"
test "$(sha256sum "${coefficient_source}" | awk '{print $1}')" = "${expected_coeff}"
test "$(sha256sum "${irisk_library}" | awk '{print $1}')" = "${expected_irisk}"
test "$(sha256sum "${HISTORY_FILE}" | awk '{print $1}')" = "${expected_history}"
test "$(sha256sum "${RESTART_FILE}" | awk '{print $1}')" = "${expected_restart}"
test ! -e "${RUN_ROOT}"

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
export ATHENA_Z4C_LEAN_RUNTIME=1
export ATHENA_AMR_HISTORY_COMPATIBLE_SOURCE_ID=athena-0.1-git-f8303c6be7eb214fa1e91b646123ee0d434b3698

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
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" "${input}" \
    "${coefficient_source}" "${irisk_library}" "${HISTORY_FILE}" \
    "${RESTART_FILE}"
  module list
  sycl-ls
} > "${root}/evidence/provenance.txt" 2>&1

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  mkdir -p "${root}/smoke"
  smoke_input=${source_root}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
  (cd "${root}/smoke" && mpiexec -n 1 -ppn 1 --depth 8 --cpu-bind depth \
    "${tile_wrapper}" "${athena}" -i "${smoke_input}" \
    job/basename=vc_lean_smoke time/nlim=1 z4c/lean_runtime=true \
    > stdout.log 2> stderr.log)
fi

total_ranks=$((RUN_NODES * RANKS_PER_NODE))
basename=n512_reference_shock_optimized
command=(mpiexec -n "${total_ranks}" -ppn "${RANKS_PER_NODE}" --depth 8
  --cpu-bind depth "${tile_wrapper}"
  "${athena}" -r "${RESTART_FILE}" -t "${ATHENA_WALL_LIMIT}"
  mesh/nx1=256 mesh/nx2=512 meshblock/nx1=64 meshblock/nx2=64
  mesh_refinement/max_nmb_per_rank=2048
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1
  z4c/lapse_shock_avoiding=true z4c/lapse_shock_avoiding_kappa=1
  z4c/telegraph_lapse=false z4c/shift_mode=prescribed_zero
  z4c/shift_invariant_diagnostic=false z4c/diss=0.50
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

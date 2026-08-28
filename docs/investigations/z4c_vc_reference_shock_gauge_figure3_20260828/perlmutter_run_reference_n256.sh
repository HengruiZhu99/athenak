#!/usr/bin/env bash
set -euo pipefail

# Run inside a granted Perlmutter A100 allocation. Every invocation owns one
# immutable segment; continuations consume an explicit restart and append to
# the same record-mode AMR authority.
: "${RUN_ROOT:?fresh absolute segment directory}"
: "${HISTORY_FILE:?absolute fresh/shared record authority}"
: "${RUN_TLIM:?coordinate-time target}"

campaign=${CAMPAIGN_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828}
source_root=${SOURCE_ROOT:-${campaign}/source}
build_root=${BUILD_ROOT:-${campaign}/build-cuda-mpi}
coefficient_source=${COEFFICIENT_SOURCE:-${campaign}/brill_global_128x32.coefficients}
profile=${PERLMUTTER_PROFILE:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh}
python_bin=${NERSC_PYTHON:-/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3}
irisk_library=${IRISK_LIBRARY:-/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823/source/irisk-authority/build/serial-gcc/src/libiris_athenak_interpolator.a}

expected_source=00e66dfa5e0e7f7f2c711166998806a05decbd55
expected_tree=578bfd4ede74ddc7464f82e4c7b76f4111a4ad76
expected_exe=c5153a5c25c4b2aba22737061baa00628badcd6eade2c66461ac182708677e55
expected_cache=273fe5f79f18d39f19748aff573c1ac204ebe9aae14f61533ca99c42a3c13f8a
expected_input=6c694cf871a3d694d745f0fb58b279b6cd07516463ac8ad54f1c91d2689c90ba
expected_coeff=1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10
expected_irisk=d4afad6d3a20a8dd8197eb7d70d5a23903a7e2401a5d8b034d32005bf07f3f39

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
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
test -r "${profile}"
test ! -e "${RUN_ROOT}"
if [[ -z "${RESTART_FILE:-}" ]]; then
  test ! -e "${HISTORY_FILE}"
else
  test -s "${RESTART_FILE}"
  test -s "${HISTORY_FILE}"
fi

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
cp "${input}" "${RUN_ROOT}/"
cp "${coefficient_source}" "${RUN_ROOT}/brill_global_128x32.coefficients"

export COLLAPSE_ROOT=${campaign}
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
    "${root}/brill_vc_reference_shock_gauge.athinput" \
    "${root}/brill_global_128x32.coefficients" "${irisk_library}"
  sed -n '1,10p' "${root}/brill_global_128x32.coefficients"
  module list
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${root}/evidence/provenance.txt" 2>&1

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits --loop=1 > "${root}/evidence/gpu-memory.csv" 2>&1 &
monitor_pid=$!
finish() {
  code=$?
  trap - EXIT
  set +e
  kill "${monitor_pid}" 2>/dev/null
  wait "${monitor_pid}" 2>/dev/null
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

restart=(-i brill_vc_reference_shock_gauge.athinput)
if [[ -n "${RESTART_FILE:-}" ]]; then
  restart=(-r "${RESTART_FILE}")
  sha256sum "${RESTART_FILE}" > "${root}/evidence/restart-input.sha256"
fi

wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${root}/rank-bindings" --require-cuda -- "${athena}"
  "${restart[@]}" -t 01:40:00
  mesh_refinement/amr_history_mode=record
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1
  z4c/lapse_shock_avoiding=true z4c/lapse_shock_avoiding_kappa=1
  z4c/telegraph_lapse=false z4c/shift_mode=prescribed_zero
  z4c/shift_invariant_diagnostic=true z4c/diss=0.50
  z4c/damp_kappa1=0 z4c/damp_kappa2=0
  job/basename=n256_reference_shock_record
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file=n256_reference_shock_record-constraints.dat)
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

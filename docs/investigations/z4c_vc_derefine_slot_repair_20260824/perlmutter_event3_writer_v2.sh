#!/usr/bin/env bash
set -euo pipefail

: "${CASE_LABEL:?n128, n256, or n512}"
: "${SOURCE_ROOT:?detached AthenaK source at the compiled revision}"
: "${BUILD_ROOT:?CUDA MPI build root}"
: "${RUN_ROOT:?fresh output directory}"
: "${EXPECTED_EXE_SHA:?SHA-256 of the qualified CUDA executable}"

expected_source=6dd20656a305f2543bbbd7001550c6ac67019180
expected_history_source=athena-0.1-git-ba7daebccf337d3157442aec2125b9301308b2a8
expected_coeff=1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10
expected_history=ce3cdea1a8d0465a7c19e4ac1134ce474d8908b5f5cb12be6f20110d12e9c851

case "${CASE_LABEL}" in
  n128)
    root_nx1=64; root_nx2=128; mb_nx1=16; mb_nx2=16
    max_nmb=4096; event3_cycle=57
    ;;
  n256)
    root_nx1=128; root_nx2=256; mb_nx1=32; mb_nx2=32
    max_nmb=4096; event3_cycle=111
    ;;
  n512)
    root_nx1=256; root_nx2=512; mb_nx1=64; mb_nx2=64
    max_nmb=2048; event3_cycle=229
    ;;
  *) printf 'unsupported CASE_LABEL=%s\n' "${CASE_LABEL}" >&2; exit 2 ;;
esac

test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${expected_source}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1)"
athena="${BUILD_ROOT}/src/athena"
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${EXPECTED_EXE_SHA}"

authority_root="${SOURCE_ROOT}/docs/investigations/z4c_vc_figure3_convergence_20260823"
input="${authority_root}/brill_vc_figure3.athinput"
history="${authority_root}/evidence/authority/n256_amr_history.jsonl"
coeff="${SOURCE_ROOT}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/evidence/brill_global_128x32_origin_control.coefficients"
test "$(sha256sum "${history}" | awk '{print $1}')" = "${expected_history}"
test "$(sha256sum "${coeff}" | awk '{print $1}')" = "${expected_coeff}"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
cp "${input}" "${RUN_ROOT}/brill_vc_figure3.athinput"
cp "${history}" "${RUN_ROOT}/n256_amr_history.jsonl"
cp "${coeff}" "${RUN_ROOT}/brill_global_128x32.coefficients"

# Capture only the authoritative third topology transaction.  The per-parent
# writer follows every local derefinement family from the independent injection
# oracle through A16, then records the first post-event RHS and RK update.
export ATHENA_Z4C_VC_AMR_LIFECYCLE=all
export ATHENA_Z4C_VC_AMR_LIFECYCLE_CYCLE="${event3_cycle}"
export ATHENA_Z4C_VC_AMR_LIFECYCLE_JSONL="${RUN_ROOT}/vc_amr_lifecycle_event3.jsonl"
export ATHENA_Z4C_VC_DEREFINE_WRITER_JSONL="${RUN_ROOT}/vc_derefine_writer_event3.jsonl"
export ATHENA_Z4C_VC_DEREFINE_WRITER_CYCLE="${event3_cycle}"
export ATHENA_Z4C_VC_DEREFINE_SLOT_AUDIT="${RUN_ROOT}/vc_derefine_slot_audit.json"
export ATHENA_AMR_HISTORY_COMPATIBLE_SOURCE_ID="${expected_history_source}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta

wrapper="${SOURCE_ROOT}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py"
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${RUN_ROOT}/rank-bindings" --require-cuda -- "${athena}"
  -i brill_vc_figure3.athinput -t 00:20:00
  mesh/nx1="${root_nx1}" mesh/nx2="${root_nx2}"
  meshblock/nx1="${mb_nx1}" meshblock/nx2="${mb_nx2}"
  mesh_refinement/max_nmb_per_rank="${max_nmb}"
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file=n256_amr_history.jsonl
  time/tlim=0.31 time/nlim=-1 job/basename="${CASE_LABEL}_event3_writer"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${CASE_LABEL}_event3_writer-constraints.dat")

printf '%q ' "${command[@]}" > "${RUN_ROOT}/command.txt"
printf '\n' >> "${RUN_ROOT}/command.txt"
env | LC_ALL=C sort > "${RUN_ROOT}/evidence/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${RUN_ROOT}/evidence/slurm-job.txt"
{
  git -C "${SOURCE_ROOT}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${BUILD_ROOT}/CMakeCache.txt" \
    "${RUN_ROOT}/brill_vc_figure3.athinput" \
    "${RUN_ROOT}/n256_amr_history.jsonl" \
    "${RUN_ROOT}/brill_global_128x32.coefficients"
  module list
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${RUN_ROOT}/evidence/provenance.txt" 2>&1

set +e
(cd "${RUN_ROOT}" && "${command[@]}" > stdout.log 2> stderr.log)
status=$?
set -e
printf '%s\n' "${status}" > "${RUN_ROOT}/athena-exit"
find "${RUN_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${RUN_ROOT}/SHA256SUMS"
(cd "${RUN_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
exit "${status}"

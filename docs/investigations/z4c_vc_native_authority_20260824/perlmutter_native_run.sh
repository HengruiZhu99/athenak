#!/usr/bin/env bash
set -euo pipefail

: "${CASE_LABEL:?n128, n256, or n512}"
: "${AMR_HISTORY_MODE:?record or replay}"
: "${RUN_ROOT:?fresh absolute run directory}"
: "${AUTHORITY_FILE:?absolute authority file for replay, or output basename for record}"

source_root=${SOURCE_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-derefine-slot-repair-20260824/source}
build_root=${BUILD_ROOT:-/pscratch/sd/h/hzhu/z4c-vc-native-authority-20260824/build-cuda-mpi-deterministic-history}
expected_source=d63519328214a6315a9cc1f7d5e4a1aa4bca21b0
expected_tree=9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
input_root=${source_root}/docs/investigations/z4c_vc_figure3_convergence_20260823
coefficient_source=${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/evidence/brill_global_128x32_origin_control.coefficients
run_tlim=${RUN_TLIM:-2.5}

test -r "${profile}"
export COLLAPSE_ROOT=${COLLAPSE_ROOT:-/pscratch/sd/h/hzhu/collapse-critical-perlmutter}
# shellcheck source=/dev/null
source "${profile}"
test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test -x "${build_root}/src/athena"
test ! -e "${RUN_ROOT}"

case "${CASE_LABEL}" in
  n128)
    root_nx1=64; root_nx2=128; mb_nx1=16; mb_nx2=16; max_nmb=4096
    ;;
  n256)
    root_nx1=128; root_nx2=256; mb_nx1=32; mb_nx2=32; max_nmb=4096
    ;;
  n512)
    root_nx1=256; root_nx2=512; mb_nx1=64; mb_nx2=64; max_nmb=2048
    ;;
  *)
    printf 'unsupported CASE_LABEL=%s\n' "${CASE_LABEL}" >&2
    exit 2
    ;;
esac

case "${AMR_HISTORY_MODE}" in
  record)
    test "${CASE_LABEL}" = n256
    history_argument=${AUTHORITY_FILE}
    ;;
  replay)
    test -f "${AUTHORITY_FILE}"
    history_argument=n256_native_authority.jsonl
    ;;
  *)
    printf 'unsupported AMR_HISTORY_MODE=%s\n' "${AMR_HISTORY_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/rank-bindings"
cp "${input_root}/brill_vc_figure3.athinput" "${RUN_ROOT}/"
cp "${coefficient_source}" "${RUN_ROOT}/brill_global_128x32.coefficients"
if test "${AMR_HISTORY_MODE}" = replay; then
  cp "${AUTHORITY_FILE}" "${RUN_ROOT}/${history_argument}"
fi

athena=${build_root}/src/athena
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta

scontrol show job "${SLURM_JOB_ID}" > "${RUN_ROOT}/evidence/slurm-job.txt"
env | LC_ALL=C sort > "${RUN_ROOT}/evidence/environment.txt"
{
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
    "${RUN_ROOT}/brill_vc_figure3.athinput" \
    "${RUN_ROOT}/brill_global_128x32.coefficients"
  module list
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${RUN_ROOT}/evidence/provenance.txt" 2>&1

command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${RUN_ROOT}/rank-bindings" --require-cuda -- "${athena}"
  -i brill_vc_figure3.athinput -t 00:20:00
  mesh/nx1="${root_nx1}" mesh/nx2="${root_nx2}"
  meshblock/nx1="${mb_nx1}" meshblock/nx2="${mb_nx2}"
  mesh_refinement/max_nmb_per_rank="${max_nmb}"
  mesh_refinement/amr_history_mode="${AMR_HISTORY_MODE}"
  mesh_refinement/amr_history_file="${history_argument}"
  time/tlim="${run_tlim}" time/nlim=-1 job/basename="${CASE_LABEL}_native_${AMR_HISTORY_MODE}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${CASE_LABEL}_native_${AMR_HISTORY_MODE}-constraints.dat")

printf '%q ' "${command[@]}" > "${RUN_ROOT}/command.txt"
printf '\n' >> "${RUN_ROOT}/command.txt"
set +e
(cd "${RUN_ROOT}" && "${command[@]}" > stdout.log 2> stderr.log)
status=$?
set -e
printf '%d\n' "${status}" > "${RUN_ROOT}/athena-exit"
find "${RUN_ROOT}" -type f ! -name SHA256SUMS -print0 |
  LC_ALL=C sort -z | xargs -0r sha256sum > "${RUN_ROOT}/SHA256SUMS"
(cd "${RUN_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
exit "${status}"

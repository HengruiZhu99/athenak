#!/usr/bin/env bash
set -euo pipefail

: "${CAMPAIGN_ROOT:?fresh campaign root required}"
: "${SOURCE_ROOT:?}"
: "${BUILD_ROOT:?}"
: "${REPLAY_INPUT:?}"
: "${COEFFICIENT_FILE:?}"
: "${AUTHORITY_FILE:?}"

expected_source=449dac7b947f22d373f096dc223eeda476582580
expected_tree=be23a7348a043094ddcf2dcaed538fbb929f688b
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
expected_exe=a3c6894079c030f66c0e576a43ea640b6191e375eb1acb3eeb575215bd19c54d
expected_cache=c06022bad22e291a897612b6466d9ee1a6284541b6d15dcfcebff1f1a39a2009
expected_coeff=ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
expected_authority=874551cc68e7dab4d40b854b31ab6b42aff9d2eae0ca9faf5985c41ef14a589f

test ! -e "${CAMPAIGN_ROOT}"
case "${SLURM_JOB_QOS:-}" in
  shared_interactive|gpu_shared_interactive|gpu_shared_interactive_ss11) ;;
  *) printf 'unexpected QOS: %s\n' "${SLURM_JOB_QOS:-<missing>}" >&2; exit 2 ;;
esac
test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 2
test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_GPUS_PER_NODE:-}" = 1

test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD^{tree})" = "${expected_tree}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)"
test "$(git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"

athena="${BUILD_ROOT}/src/athena"
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${BUILD_ROOT}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"
test "$(sha256sum "${COEFFICIENT_FILE}" | awk '{print $1}')" = "${expected_coeff}"
test "$(sha256sum "${AUTHORITY_FILE}" | awk '{print $1}')" = "${expected_authority}"

profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper="${SOURCE_ROOT}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py"
export COLLAPSE_ROOT="${CAMPAIGN_ROOT}"
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU

mkdir -p "${CAMPAIGN_ROOT}/evidence"
root="$(cd "${CAMPAIGN_ROOT}" && pwd)"
cp "${REPLAY_INPUT}" "${root}/brill_o4_common_tree.athinput"
cp "${COEFFICIENT_FILE}" "${root}/brill_global_48x32.coefficients"
cp "${AUTHORITY_FILE}" "${root}/n256_amr_history.jsonl"

finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s\n' "${code}" > "${root}/orchestration-status"
  git -C "${SOURCE_ROOT}" status --porcelain=v1 > "${root}/evidence/source-status.final"
  sacct -j "${SLURM_JOB_ID}" -X -n -P \
    -o JobID,JobName,State,ExitCode,Elapsed,NodeList,ReqTRES,AllocTRES \
    > "${root}/evidence/sacct.txt"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
    -print0 | LC_ALL=C sort -z | xargs -0r sha256sum > "${root}/SHA256SUMS"
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
    "${root}/brill_o4_common_tree.athinput" \
    "${root}/brill_global_48x32.coefficients" \
    "${root}/n256_amr_history.jsonl"
  module list
} > "${root}/evidence/provenance.txt" 2>&1

run_case() {
  local label=$1 nx1=$2 nx2=$3 mb1=$4 mb2=$5 capacity=$6
  local case_root="${root}/common_tree/${label}"
  local bindings="${case_root}/rank-bindings"
  test ! -e "${case_root}"
  mkdir -p "${bindings}"
  local command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1
    --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=map_gpu:0
    --cpu-bind=cores --exact --kill-on-bad-exit=1
    "${python_bin}" "${wrapper}" --evidence-dir "${bindings}"
    --require-cuda -- "${athena}" -i "${root}/brill_o4_common_tree.athinput"
    mesh/nx1="${nx1}" mesh/nx2="${nx2}"
    meshblock/nx1="${mb1}" meshblock/nx2="${mb2}"
    time/tlim=5.0 time/nlim=-1 job/basename="${label}"
    problem/brill_global_coefficients_file="${root}/brill_global_48x32.coefficients"
    problem/constraint_summary_file="${case_root}/${label}.constraints.dat"
    z4c/grid_centering=vertex
    mesh_refinement/max_nmb_per_rank="${capacity}"
    mesh_refinement/amr_history_mode=replay
    mesh_refinement/amr_history_file="${root}/n256_amr_history.jsonl"
    mesh_refinement/amr_history_topology_only_centering_compatibility=cell_to_vertex
    mesh_refinement/amr_history_compatible_source_id=athena-0.1-git-d0d0b648bab09afb33453132075f1b813306526a)
  printf '%q ' "${command[@]}" > "${case_root}/command.txt"
  printf '\n' >> "${case_root}/command.txt"
  set +e
  (cd "${case_root}" && "${command[@]}" > stdout.log 2> stderr.log)
  status=$?
  set -e
  printf '%s\n' "${status}" > "${case_root}/exit-status"
  "${python_bin}" - "${bindings}" <<'PY'
import json, pathlib, sys
files = sorted(pathlib.Path(sys.argv[1]).glob("rank_binding_*.json"))
assert len(files) == 1, files
record = json.loads(files[0].read_text())
assert record["binding_verified"] is True, record
assert "NVIDIA A100-SXM4" in record["gpu_name"], record
assert record["selected_uuid"], record
PY
  if [[ ${status} -eq 0 ]] && grep -Fq 'Terminating on time limit' "${case_root}/stdout.log"; then
    printf 'REACHED_TLIM\n' > "${case_root}/disposition"
    return 0
  fi
  printf 'FAIL_CLOSED\n' > "${case_root}/disposition"
  return 1
}

overall=0
run_case n256 128 256 32 32 512 || overall=1
run_case n512 256 512 64 64 256 || overall=1
exit "${overall}"

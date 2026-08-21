#!/usr/bin/env bash
set -euo pipefail

: "${CAMPAIGN_ROOT:?}"
: "${SOURCE_ROOT:?}"
: "${EXPECTED_SOURCE_COMMIT:?}"
: "${EXPECTED_BUILD_SOURCE_COMMIT:?}"
: "${BUILD_ROOT:?}"
: "${EXPECTED_EXE_SHA256:?}"
: "${EXPECTED_CACHE_SHA256:?}"
: "${INPUT_FILE:?}"
: "${EXPECTED_INPUT_SHA256:?}"
: "${COEFFICIENT_FILE:?}"
: "${EXPECTED_COEFFICIENT_SHA256:?}"
: "${SEGMENT_ROOT:?fresh segment root required}"
: "${CASE_LABEL:?n128, n256, or n512}"
: "${HISTORY_MODE:?record or replay}"
: "${HISTORY_FILE:?}"
: "${ROOT_NX1:?}"
: "${ROOT_NX2:?}"
: "${MB_NX1:?}"
: "${MB_NX2:?}"
: "${MAX_NMB_PER_RANK:?}"
: "${RUN_TLIM:?}"

test ! -e "${SEGMENT_ROOT}"
case "${SLURM_JOB_QOS:-}" in
  shared_interactive|gpu_shared_interactive|gpu_shared_interactive_ss11) ;;
  *) printf 'unexpected Slurm QOS: %s\n' "${SLURM_JOB_QOS:-<missing>}" >&2; exit 2 ;;
esac
test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_GPUS_PER_NODE:-}" = 1

test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)"
test "$(git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD)" = \
  6739bc623081648af9e752b616d9671527922cbf
while IFS= read -r changed; do
  case "${changed}" in
    docs/investigations/brill_o4_dchi001_replay_convergence_20260821/*) ;;
    *) printf 'post-build change outside campaign tooling: %s\n' "${changed}" >&2; exit 2 ;;
  esac
done < <(git -C "${SOURCE_ROOT}" diff --name-only \
  "${EXPECTED_BUILD_SOURCE_COMMIT}" "${EXPECTED_SOURCE_COMMIT}")

athena="${BUILD_ROOT}/src/athena"
test -x "${athena}"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${EXPECTED_EXE_SHA256}"
test "$(sha256sum "${BUILD_ROOT}/CMakeCache.txt" | awk '{print $1}')" = \
  "${EXPECTED_CACHE_SHA256}"
test "$(sha256sum "${INPUT_FILE}" | awk '{print $1}')" = \
  "${EXPECTED_INPUT_SHA256}"
test "$(sha256sum "${COEFFICIENT_FILE}" | awk '{print $1}')" = \
  "${EXPECTED_COEFFICIENT_SHA256}"
case "${CASE_LABEL}:${HISTORY_MODE}:${ROOT_NX1}:${ROOT_NX2}:${MB_NX1}:${MB_NX2}:${MAX_NMB_PER_RANK}" in
  n128:replay:64:128:16:16:16384) ;;
  n256:record:128:256:32:32:16384) ;;
  n512:replay:256:512:64:64:16384) ;;
  *) printf 'unsupported frozen Perlmutter run contract\n' >&2; exit 2 ;;
esac
if [[ "${HISTORY_MODE}" = record ]]; then
  if [[ -z "${RESTART_FILE:-}" ]]; then
    test ! -e "${HISTORY_FILE}"
  else
    : "${EXPECTED_HISTORY_SHA256:?}"
    test "$(sha256sum "${HISTORY_FILE}" | awk '{print $1}')" = \
      "${EXPECTED_HISTORY_SHA256}"
  fi
else
  : "${EXPECTED_HISTORY_SHA256:?}"
  test "$(sha256sum "${HISTORY_FILE}" | awk '{print $1}')" = \
    "${EXPECTED_HISTORY_SHA256}"
fi

profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper="${SOURCE_ROOT}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py"
export COLLAPSE_ROOT="${CAMPAIGN_ROOT}"
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU

run_one_gpu() {
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 \
    --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
    --kill-on-bad-exit=1 "$@"
}

if [[ "${RUN_DEVICE_QUALIFICATION:-0}" = 1 ]]; then
  qualification="${CAMPAIGN_ROOT}/qualification-perlmutter"
  test ! -e "${qualification}"
  mkdir -p "${qualification}/state-extraction" "${qualification}/short"
  regex='athena[.](amr_history_format|amr_history_extension_static|amr_history_shadow_static|amr_history_integration|z4c_state_admissibility|z4c_state_admissibility_static|z4c_chi_prolongation|z4c_cartoon_amr_transfer_qualification|z4c_cartoon_axis_centered_derivatives|z4c_coarse_cache_ownership|z4c_timestep_contract|z4c_amr_configuration_static)$'
  run_one_gpu ctest --test-dir "${BUILD_ROOT}" --output-on-failure \
    -R "${regex}" > "${qualification}/focused-ctest.log" 2>&1
  grep -F '100% tests passed, 0 tests failed out of 12' \
    "${qualification}/focused-ctest.log" >/dev/null

  set +e
  (cd "${qualification}/state-extraction" && \
    run_one_gpu env ATHENA_TEST_Z4C_STATE_EXTRACTION=selected_negative_chi \
      "${athena}" -i "${INPUT_FILE}" time/nlim=0 \
      mesh_refinement/max_nmb_per_rank=128 job/basename=state_extraction \
      problem/brill_global_coefficients_file="${COEFFICIENT_FILE}" \
      > stdout.log 2> stderr.log)
  state_status=$?
  set -e
  test "${state_status}" -ne 0
  "${python_bin}" - "${qualification}/state-extraction/z4c_state_failure.json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["schema"] == "z4c_state_admissibility_v1"
assert d["reason"] == "nonpositive_chi" and d["chi"] == -1
assert d["checkpoint"] == "PRE_RHS" and d["rk_stage"] == 0
assert len(d["state25"]) == 25 and len(d["logical_location"]) == 4
assert "relative_level" in d and "spd_pivots" in d
PY

  short="${qualification}/short"
  short_history="${short}/authority.jsonl"
  short_case() {
    local name=$1; shift
    mkdir -p "${short}/${name}"
    (cd "${short}/${name}" && run_one_gpu "${athena}" -i "${INPUT_FILE}" \
      problem/brill_global_coefficients_file="${COEFFICIENT_FILE}" \
      mesh_refinement/max_nmb_per_rank=1024 output2/dcycle=4 \
      job/basename="${name}" "$@" > stdout.log 2> stderr.log)
  }
  short_case n256 mesh/nx1=128 mesh/nx2=256 meshblock/nx1=32 meshblock/nx2=32 \
    mesh_refinement/amr_history_mode=record \
    mesh_refinement/amr_history_file="${short_history}" time/nlim=8
  authority_restart=$(find "${short}/n256/rst" -type f -name '*.00001.rst' -print -quit)
  test -n "${authority_restart}"
  (cd "${short}/n256" && run_one_gpu "${athena}" -r "${authority_restart}" \
    time/nlim=12 output2/dcycle=4 > continued.log 2> continued.err)
  short_case n128 mesh/nx1=64 mesh/nx2=128 meshblock/nx1=16 meshblock/nx2=16 \
    mesh_refinement/amr_history_mode=replay \
    mesh_refinement/amr_history_file="${short_history}" time/nlim=8
  n128_restart=$(find "${short}/n128/rst" -type f -name '*.00001.rst' -print -quit)
  test -n "${n128_restart}"
  (cd "${short}/n128" && run_one_gpu "${athena}" -r "${n128_restart}" \
    time/nlim=12 output2/dcycle=4 > continued.log 2> continued.err)
  short_case n512 mesh/nx1=256 mesh/nx2=512 meshblock/nx1=64 meshblock/nx2=64 \
    mesh_refinement/amr_history_mode=replay \
    mesh_refinement/amr_history_file="${short_history}" time/nlim=8
  "${python_bin}" - "${short}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
events = [json.loads(x) for x in (root / "authority.jsonl").read_text().splitlines()]
events = [x for x in events if x.get("type") == "event"]
assert len(events) >= 2
for name in ("n128", "n512"):
    ledger = next((root / name).glob("*.amr_history_replay.jsonl"))
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert rows and all(x["exact_match"] for x in rows)
    assert all(abs(x["ulp_difference"]) <= 1 for x in rows)
    expected = [x["tree_checksum"] for x in events[1:len(rows) + 1]]
    assert [x["tree_checksum"] for x in rows] == expected
PY
  sha256sum "${athena}" "${BUILD_ROOT}/CMakeCache.txt" \
    "${qualification}/focused-ctest.log" \
    "${qualification}/state-extraction/z4c_state_failure.json" \
    "${short_history}" > "${qualification}/SHA256SUMS"
  (cd "${qualification}" && sha256sum -c SHA256SUMS >/dev/null)
  printf 'PERLMUTTER_CUDA_O4_REPLAY_QUALIFICATION_PASS\n' > \
    "${qualification}/status"
else
  qualification="${CAMPAIGN_ROOT}/qualification-perlmutter"
  grep -Fx PERLMUTTER_CUDA_O4_REPLAY_QUALIFICATION_PASS \
    "${qualification}/status" >/dev/null
  (cd "${qualification}" && sha256sum -c SHA256SUMS >/dev/null)
fi

mkdir -p "${SEGMENT_ROOT}/evidence" "${SEGMENT_ROOT}/rank-bindings"
root="$(cd "${SEGMENT_ROOT}" && pwd)"
cp "${INPUT_FILE}" "${root}/brill_o4_common_tree.athinput"
cp "${COEFFICIENT_FILE}" "${root}/brill_global_48x32.coefficients"

finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s\n' "${code}" > "${root}/orchestration-status"
  git -C "${SOURCE_ROOT}" status --porcelain=v1 > \
    "${root}/evidence/source-status.final"
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
  printf 'SOURCE_COMMIT=%s\nBUILD_SOURCE_COMMIT=%s\n' \
    "${EXPECTED_SOURCE_COMMIT}" "${EXPECTED_BUILD_SOURCE_COMMIT}"
  git -C "${SOURCE_ROOT}" rev-parse HEAD HEAD^{tree}
  git -C "${SOURCE_ROOT}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${BUILD_ROOT}/CMakeCache.txt" \
    "${root}/brill_o4_common_tree.athinput" \
    "${root}/brill_global_48x32.coefficients"
  module list
} > "${root}/evidence/provenance.txt" 2>&1

restart_args=(-i brill_o4_common_tree.athinput)
if [[ -n "${RESTART_FILE:-}" ]]; then
  : "${RESTART_SHA256:?}"
  test "$(sha256sum "${RESTART_FILE}" | awk '{print $1}')" = \
    "${RESTART_SHA256}"
  restart_args=(-r "${RESTART_FILE}")
  sha256sum "${RESTART_FILE}" > "${root}/evidence/restart-input.sha256"
fi

command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1
  --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=map_gpu:0
  --cpu-bind=cores --exact --kill-on-bad-exit=1 --time=01:45:00
  "${python_bin}" "${wrapper}" --evidence-dir "${root}/rank-bindings"
  --require-cuda -- "${athena}" "${restart_args[@]}"
  mesh/nx1="${ROOT_NX1}" mesh/nx2="${ROOT_NX2}"
  meshblock/nx1="${MB_NX1}" meshblock/nx2="${MB_NX2}"
  mesh_refinement/max_nmb_per_rank="${MAX_NMB_PER_RANK}"
  mesh_refinement/amr_history_mode="${HISTORY_MODE}"
  mesh_refinement/amr_history_file="${HISTORY_FILE}"
  time/tlim="${RUN_TLIM}" time/nlim=-1
  output2/dcycle=512 output3/dcycle=128 output4/dcycle=128
  output5/dcycle=0 output6/dcycle=128 output7/dcycle=0 output8/dcycle=0
  job/basename="${CASE_LABEL}"
  problem/brill_global_coefficients_file=brill_global_48x32.coefficients
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
assert record["rank"] == 0 and record["local_rank"] == 0, record
assert record["binding_verified"] is True, record
assert "NVIDIA A100-SXM4" in record["gpu_name"], record
assert record["selected_uuid"], record
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
if [[ -e "${HISTORY_FILE}" ]]; then
  sha256sum "${HISTORY_FILE}" > "${root}/evidence/history-after.sha256"
fi
if [[ "${disposition}" = UNCLASSIFIED_CLEAN_EXIT ]]; then exit 2; fi
exit "${status}"

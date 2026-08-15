#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${bundle_dir}/.." && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
test "${root}" = "${REMOTE_ROOT}"
test "${SLURM_JOB_NAME:-}" = "${EXPECTED_JOB_NAME}"
test "${SLURM_JOB_NUM_NODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_GPUS_PER_NODE:-}" = 1
case "${SLURM_JOB_QOS:-}" in
  shared_interactive|gpu_shared_interactive|gpu_shared_interactive_ss11) ;;
  *) printf 'unexpected shared QOS %s\n' "${SLURM_JOB_QOS:-unset}" >&2; exit 2 ;;
esac
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${root}/run"

run_root=${root}/run
case_root=${run_root}/cases/${CASE_NAME}
evidence=${run_root}/evidence
mkdir -p "${case_root}/bindings" "${evidence}"
finalize() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run_root}/exit-status.txt"
  git -C "${SOURCE_PATH}" status --porcelain=v1 > "${evidence}/source-status.final"
  exit "${status}"
}
trap finalize EXIT

export COLLAPSE_ROOT="${root}"
source "${PROFILE_PATH}"
export OMP_NUM_THREADS=32 KOKKOS_NUM_THREADS=32
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${evidence}/environment.txt"
module -t list > "${evidence}/modules.txt" 2>&1
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence}/hosts.txt"
scontrol show node -o "${SLURM_JOB_NODELIST}" > "${evidence}/node-profile.txt"
grep -Eq '(^|,)hbm80g(,|$)' < <(sed -n 's/.*ActiveFeatures=\([^ ]*\).*/\1/p' \
  "${evidence}/node-profile.txt")
! grep -Eq '(^|,)hbm40g(,|$)' < <(sed -n 's/.*ActiveFeatures=\([^ ]*\).*/\1/p' \
  "${evidence}/node-profile.txt")

test "$(git -C "${SOURCE_PATH}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
test "$(git -C "${SOURCE_PATH}" rev-parse HEAD^{tree})" = "${EXPECTED_SOURCE_TREE}"
test "$(git -C "${SOURCE_PATH}/kokkos" rev-parse HEAD)" = "${EXPECTED_SOURCE_KOKKOS}"
test -z "$(git -C "${SOURCE_PATH}" status --short)"
test "$(sha256sum "${EXECUTABLE}" | awk '{print $1}')" = "${EXPECTED_EXECUTABLE_SHA256}"
test "$(sha256sum "${bundle_dir}/brill_global_48x32.coefficients" | awk '{print $1}')" = \
  "${EXPECTED_COEFFICIENT_SHA256}"
test "$(sha256sum "${V7_ROOT}/SHA256SUMS" | awk '{print $1}')" = "${EXPECTED_V7_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${V7_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = "${EXPECTED_V7_DETACHED_SHA256}"
test "$(sha256sum "${V7_ROOT}/allocation/sacct-settled.psv" | awk '{print $1}')" = "${EXPECTED_V7_SACCT_SHA256}"
test "$(sha256sum "${V7_ROOT}/run/comparison.json" | awk '{print $1}')" = "${EXPECTED_V7_COMPARISON_SHA256}"
mapfile -t v7_histories < <(find "${V7_ROOT}/run/cases/${V7_ZERO_CASE}" \
  -maxdepth 1 -type f -name '*.hst' -print)
test "${#v7_histories[@]}" -eq 1 && test ! -L "${v7_histories[0]}"
test "$(sha256sum "${v7_histories[0]}" | awk '{print $1}')" = \
  "${EXPECTED_V7_ZERO_HISTORY_SHA256}"
test "$(sha256sum "${V7_ROOT}/run/cases/${V7_ZERO_CASE}/run.log" | awk '{print $1}')" = "${EXPECTED_V7_ZERO_LOG_SHA256}"
restart=${V7_ROOT}/run/cases/${V7_ZERO_CASE}/rst/${RESTART_BASENAME}
test -f "${restart}" && test ! -L "${restart}"
test "$(stat -c '%s' "${restart}")" = "${EXPECTED_RESTART_BYTES}"
test "$(sha256sum "${restart}" | awk '{print $1}')" = "${EXPECTED_RESTART_SHA256}"
sha256sum "${V7_ROOT}/SHA256SUMS" "${V7_ROOT}/SHA256SUMS.sha256" \
  "${V7_ROOT}/allocation/sacct-settled.psv" "${V7_ROOT}/run/comparison.json" \
  "${restart}" "${EXECUTABLE}" > "${evidence}/prerequisites.sha256"

wrapper=${SOURCE_PATH}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 --time=03:45:00
  "${PYTHON_BIN}" "${wrapper}" --evidence-dir "${case_root}/bindings"
  --require-cuda -- "${EXECUTABLE}" -r "${restart}" -d "${case_root}"
  job/basename="${OUTPUT_BASENAME}"
  problem/brill_global_coefficients_file="${bundle_dir}/brill_global_48x32.coefficients"
  time/tlim="${TARGET_TLIM}" time/nmb_total_limit="${EXPECTED_NMB_TOTAL_LIMIT}"
  mesh_refinement/max_nmb_per_rank="${EXPECTED_MAX_NMB_PER_RANK}"
  mesh_refinement/num_levels="${EXPECTED_NUM_LEVELS}"
  z4c_amr/max_ref_lev="${EXPECTED_MAX_REF_LEV}"
  z4c/telegraph_lapse=true z4c/telegraph_max_K=true
  z4c/telegraph_damping_prescription=max_domain_abs_K
  z4c/telegraph_tau=1.0 z4c/telegraph_kappa=1.0
  z4c/shift_Gamma=0 z4c/shift_alpha2Gamma=0 z4c/shift_H=0
  z4c/shift_advect=0 z4c/shift_eta=0 z4c/shift_eta_max_K=false
  z4c/damp_kappa1=0.0 z4c/damp_kappa2=0.0 z4c/target_kappa1=0.0
  z4c/damp_kappa1_max_K=false z4c/roll_kappa=false z4c/floor_chi=false
  z4c/diss=0.5 z4c_amr/dchi_max=0.02)
printf '%q ' "${command[@]}" > "${case_root}/command.txt"
printf '\n' >> "${case_root}/command.txt"
set +e
"${command[@]}" > "${case_root}/run.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${case_root}/exit-status.txt"
"${PYTHON_BIN}" -B "${bundle_dir}/analyze_pair.py" case \
  --case-dir "${case_root}" --name "${CASE_NAME}" --tau 1.0 --kappa 1.0 \
  --dissipation 0.5 --shift-condition zero_shift --output "${case_root}/result.json"
"${PYTHON_BIN}" -B - "${case_root}/result.json" "${EXPECTED_GPU_NAME//_/ }" <<'PY'
import json, pathlib, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d['qualification_claim'] is False
assert len(d['rank_bindings']) == 1
assert d['rank_bindings'][0]['binding_verified'] is True
assert d['rank_bindings'][0]['gpu_name'] == sys.argv[2]
PY
test -z "$(git -C "${SOURCE_PATH}" status --short)"
printf 'BRILL_DOMAIN64_ZERO_SHIFT_RESTART_ATTEMPT_COMPLETE\n' > "${run_root}/verdict.txt"
exit "${status}"

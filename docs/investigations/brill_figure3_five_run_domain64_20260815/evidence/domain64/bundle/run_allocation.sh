#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${bundle_dir}/.." && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
run_root=${root}/run
cases_root=${run_root}/cases
evidence=${run_root}/evidence

test "${root}" = "${REMOTE_ROOT}"
test "${SLURM_JOB_NAME:-}" = "${EXPECTED_JOB_NAME}"
test "${SLURM_JOB_NUM_NODES:-}" = 1
test "${SLURM_NTASKS:-}" = "${EXPECTED_RANKS}"
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_GPUS_PER_NODE:-}" = "${EXPECTED_GPUS}"
case "${SLURM_JOB_QOS:-}" in
  shared_interactive|gpu_shared_interactive|gpu_shared_interactive_ss11) ;;
  *) printf 'unexpected shared QOS %s\n' "${SLURM_JOB_QOS:-unset}" >&2; exit 2 ;;
esac
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${run_root}"
mkdir -p "${cases_root}" "${evidence}"

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

test "$(sha256sum "${BASELINE_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_BASELINE_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${BASELINE_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_BASELINE_DETACHED_SHA256}"
(cd "${BASELINE_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null && \
  sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "${BASELINE_CASE}/brill_fig3_max_domain_abs_K_n128.z4c.user.hst" | awk '{print $1}')" = \
  "${EXPECTED_BASELINE_HISTORY_SHA256}"

test "$(sha256sum "${PRIOR_FIXED_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_ROOT_MANIFEST_SHA256}"
test "$(sha256sum "${PRIOR_FIXED_ROOT}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_DETACHED_SHA256}"
(cd "${PRIOR_FIXED_ROOT}" && sha256sum -c SHA256SUMS.sha256 >/dev/null && \
  sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "${PRIOR_FIXED_CASE}/brill_fig3_fixed_eta2_n128.z4c.user.hst" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_HISTORY_SHA256}"
test "$(sha256sum "${PRIOR_FIXED_CASE}/run.log" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_LOG_SHA256}"
test "$(sha256sum "${PRIOR_FIXED_CASE}/command.txt" | awk '{print $1}')" = \
  "${EXPECTED_PRIOR_FIXED_COMMAND_SHA256}"
bash "${bundle_dir}/verify_predecessor.sh" > \
  "${evidence}/predecessor-terminal.sha256"

sha256sum "${bundle_dir}/${INPUT_BASENAME}" \
  "${bundle_dir}/${COEFFICIENT_BASENAME}" "${EXECUTABLE}" \
  "${BASELINE_ROOT}/SHA256SUMS" "${PRIOR_FIXED_ROOT}/SHA256SUMS" \
  "${bundle_dir}/predecessor-selected.sha256" \
  "${V6_ROOT}/SHA256SUMS" "${V6_ROOT}/SHA256SUMS.sha256" > \
  "${evidence}/prerequisites.sha256"
printf '%s\n' \
  'common: one rank/one A100, cases strictly sequential; mesh_refinement/num_levels=21 z4c_amr/max_ref_lev=20 time/nmb_total_limit=16384 mesh_refinement/max_nmb_per_rank=16384 z4c/shift_eta_max_K=false' \
  'lapse: telegraph_tau=1 telegraph_kappa=1 telegraph_max_K=true' \
  'constraint damping: damp_kappa1=0 damp_kappa2=0 target_kappa1=0 damp_kappa1_max_K=false roll_kappa=false' \
  'domain: rho=[0,64] z=[-64,64], dx=0.25 preserved by nx1=256 nx2=512' \
  'case1: z4c/diss=0.5 fixed Gamma-driver eta=2' \
  'case2: z4c/diss=0.5 zero shift' > \
  "${evidence}/exact-parameter-differences.txt"

wrapper=${SOURCE_PATH}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
run_case() {
  local name=$1 basename=$2 dissipation=$3 shift_condition=$4
  local case_root=${cases_root}/${name}
  mkdir -p "${case_root}/bindings"
  local shift_args=()
  case "${shift_condition}" in
    fixed_gamma_driver_eta2)
      shift_args=(z4c/shift_Gamma=1.0 z4c/shift_eta=2.0
        z4c/shift_eta_max_K=false z4c/shift_advect=1.0)
      ;;
    zero_shift)
      shift_args=(z4c/shift_Gamma=0 z4c/shift_alpha2Gamma=0 z4c/shift_H=0
        z4c/shift_advect=0 z4c/shift_eta=0 z4c/shift_eta_max_K=false)
      ;;
    *) printf 'invalid shift condition %s\n' "${shift_condition}" >&2; return 2 ;;
  esac
  local command=(srun --nodes=1 --ntasks="${EXPECTED_RANKS}"
    --ntasks-per-node="${EXPECTED_RANKS}" --cpus-per-task=32
    --gpus-per-task=1 --gpu-bind=map_gpu:0
    --cpu-bind=cores --exact --kill-on-bad-exit=1 --time=01:55:00
    "${PYTHON_BIN}" "${wrapper}" --evidence-dir "${case_root}/bindings"
    --require-cuda -- "${EXECUTABLE}" -i "${bundle_dir}/${INPUT_BASENAME}"
    -d "${case_root}" job/basename="${basename}"
    problem/brill_global_coefficients_file="${bundle_dir}/${COEFFICIENT_BASENAME}"
    time/tlim="${TARGET_TLIM}"
    time/nmb_total_limit="${EXPECTED_NMB_TOTAL_LIMIT}"
    mesh_refinement/max_nmb_per_rank="${EXPECTED_MAX_NMB_PER_RANK}"
    mesh_refinement/num_levels="${EXPECTED_NUM_LEVELS}"
    z4c_amr/max_ref_lev="${EXPECTED_MAX_REF_LEV}"
    z4c/telegraph_lapse=true z4c/telegraph_max_K=true
    z4c/telegraph_damping_prescription=max_domain_abs_K
    z4c/telegraph_tau=1.0 z4c/telegraph_kappa=1.0
    "${shift_args[@]}"
    z4c/damp_kappa1=0.0 z4c/damp_kappa2=0.0
    z4c/target_kappa1=0.0 z4c/damp_kappa1_max_K=false
    z4c/roll_kappa=false z4c/floor_chi=false
    z4c/diss="${dissipation}" z4c_amr/dchi_max=0.02)
  printf '%q ' "${command[@]}" > "${case_root}/command.txt"
  printf '\n' >> "${case_root}/command.txt"
  set +e
  "${command[@]}" > "${case_root}/run.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "${status}" > "${case_root}/exit-status.txt"
  "${PYTHON_BIN}" -B "${bundle_dir}/analyze_pair.py" case \
    --case-dir "${case_root}" --name "${name}" --tau 1.0 \
    --kappa 1.0 --dissipation "${dissipation}" \
    --shift-condition "${shift_condition}" \
    --output "${case_root}/result.json"
  "${PYTHON_BIN}" -B - "${case_root}/result.json" "${EXPECTED_GPU_NAME//_/ }" <<'PY'
import json, pathlib, sys
data=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert len(data["rank_bindings"]) == 1
assert data["rank_bindings"][0]["binding_verified"] is True
assert data["rank_bindings"][0]["gpu_name"] == sys.argv[2]
PY
}

run_case "${CASE_KO05}" brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64_n128 0.5 fixed_gamma_driver_eta2
run_case "${CASE_ZERO_SHIFT_KO05}" brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_n128 0.5 zero_shift

"${PYTHON_BIN}" -B "${bundle_dir}/analyze_pair.py" comparison \
  --case-results "${cases_root}/${CASE_KO05}/result.json" \
  "${cases_root}/${CASE_ZERO_SHIFT_KO05}/result.json" \
  --output "${run_root}/comparison.json"
"${PYTHON_BIN}" -B - "${run_root}/comparison.json" <<'PY'
import json, pathlib, sys
data=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["all_cases_attempted"] is True
assert data["all_rank_bindings_verified"] is True
assert data["qualification_claim"] is False
assert data["case_parameter_differences"] == {
  "case1_to_case2": "fixed Gamma-driver eta=2 -> zero shift",
}
assert data["constraint_damping_control"] == {
  "z4c/damp_kappa1": 0.0,
  "z4c/damp_kappa2": 0.0,
  "z4c/target_kappa1": 0.0,
  "z4c/damp_kappa1_max_K": False,
  "z4c/roll_kappa": False,
}
PY
test -z "$(git -C "${SOURCE_PATH}" status --short)"
printf 'BRILL_FIG3_DOMAIN64_L20_TAU1_NOCD_KO05_SHIFT_PAIR_ATTEMPT_COMPLETE\n' > "${run_root}/verdict.txt"
ko05_status=$(<"${cases_root}/${CASE_KO05}/exit-status.txt")
zero_shift_status=$(<"${cases_root}/${CASE_ZERO_SHIFT_KO05}/exit-status.txt")
if (( ko05_status != 0 || zero_shift_status != 0 )); then
  exit 1
fi

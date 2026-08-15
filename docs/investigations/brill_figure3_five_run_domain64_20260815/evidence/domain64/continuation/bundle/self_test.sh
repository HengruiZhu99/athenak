#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
python_bin=${PYTHON_BIN}
if [[ ! -x "${python_bin}" ]]; then python_bin=$(command -v python3); fi
for script in allocate.sh preflight_login.sh require_bound_contract.sh \
    run_allocation.sh self_test.sh stage.sh; do
  bash -n "${bundle_dir}/${script}"
done
"${python_bin}" -B "${bundle_dir}/analyze_pair.py" self-test
test "$(sha256sum "${bundle_dir}/brill_global_48x32.coefficients" | awk '{print $1}')" = \
  "${EXPECTED_COEFFICIENT_SHA256}"
"${python_bin}" -B - "${bundle_dir}" <<'PY'
import pathlib, sys
root=pathlib.Path(sys.argv[1])
run=(root/'run_allocation.sh').read_text()
alloc=(root/'allocate.sh').read_text()
pre=(root/'preflight_login.sh').read_text()
stage=(root/'stage.sh').read_text()
assert 'cmake ' not in run
assert run.count('command=(srun ') == 1
assert '--ntasks=1 --ntasks-per-node=1 --cpus-per-task=32' in run
assert '--gpus-per-task=1 --gpu-bind=map_gpu:0' in run
assert '--time=03:45:00' in run
assert '"${EXECUTABLE}" -r "${restart}" -d "${case_root}"' in run
assert ' -i ' not in run
assert 'z4c/shift_Gamma=0' in run and 'z4c/shift_eta=0' in run
assert 'z4c/telegraph_tau=1.0 z4c/telegraph_kappa=1.0' in run
assert 'z4c/damp_kappa1=0.0 z4c/damp_kappa2=0.0' in run
assert 'z4c/damp_kappa1_max_K=false z4c/roll_kappa=false z4c/floor_chi=false' in run
assert 'z4c/diss=0.5 z4c_amr/dchi_max=0.02' in run
assert 'mesh_refinement/num_levels="${EXPECTED_NUM_LEVELS}"' in run
assert 'z4c_amr/max_ref_lev="${EXPECTED_MAX_REF_LEV}"' in run
assert 'time/nmb_total_limit="${EXPECTED_NMB_TOTAL_LIMIT}"' in run
assert 'mesh_refinement/max_nmb_per_rank="${EXPECTED_MAX_NMB_PER_RANK}"' in run
assert "--qos=shared_interactive --constraint='gpu&hbm80g'" in alloc
assert '--ntasks=1' in alloc and '--gpus-per-node=1' in alloc
assert '--time=04:00:00' in alloc
assert "steps[1][2:4] == ['TIMEOUT', '0:15']" in pre
assert "z['exit_code'] == 143" in pre
assert 'EXPECTED_RESTART_SHA256' in pre and 'EXPECTED_RESTART_BYTES' in pre
assert 'source_input.athinput' not in stage
assert 'predecessor-selected.sha256' not in stage
assert 'v1-failure.sha256' not in stage
print('BRILL_DOMAIN64_ZERO_RESTART_STATIC_SELF_TEST_PASS')
PY
if [[ -f "${bundle_dir}/bundle.sha256" ]]; then
  (cd "${bundle_dir}" && sha256sum -c bundle.sha256)
fi

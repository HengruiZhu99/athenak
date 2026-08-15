#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
test_python=${PYTHON_BIN}
if [[ ! -x "${test_python}" ]]; then
  test_python=$(command -v python3)
fi
test -x "${test_python}"

for script in allocate.sh preflight_login.sh require_bound_contract.sh \
    run_allocation.sh self_test.sh stage.sh verify_predecessor.sh; do
  bash -n "${bundle_dir}/${script}"
done
"${test_python}" -B "${bundle_dir}/analyze_pair.py" self-test

test "$(sha256sum "${bundle_dir}/${INPUT_BASENAME}" | awk '{print $1}')" = \
  "${EXPECTED_INPUT_SHA256}"
test "$(sha256sum "${bundle_dir}/${COEFFICIENT_BASENAME}" | awk '{print $1}')" = \
  "${EXPECTED_COEFFICIENT_SHA256}"

"${test_python}" -B - "${bundle_dir}" <<'PY'
import pathlib, re, sys
root=pathlib.Path(sys.argv[1])
text=(root/'run_allocation.sh').read_text()
analyzer=(root/'analyze_pair.py').read_text()
assert 'cmake ' not in text
assert text.count('run_case "${CASE_KO002}"') == 1
assert text.count('run_case "${CASE_KO05}"') == 1
assert text.count('run_case "${CASE_ZERO_SHIFT_KO05}"') == 1
assert ('run_case "${CASE_KO002}" '
        'brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko002_n128 '
        '0.02 fixed_gamma_driver_eta2') in text
assert ('run_case "${CASE_KO05}" '
        'brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko05_n128 '
        '0.5 fixed_gamma_driver_eta2') in text
assert ('run_case "${CASE_ZERO_SHIFT_KO05}" '
        'brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_n128 '
        '0.5 zero_shift') in text
assert 'mesh_refinement/num_levels="${EXPECTED_NUM_LEVELS}"' in text
assert 'z4c_amr/max_ref_lev="${EXPECTED_MAX_REF_LEV}"' in text
assert 'time/nmb_total_limit="${EXPECTED_NMB_TOTAL_LIMIT}"' in text
assert 'z4c/shift_eta_max_K=false' in text
assert '--ntasks="${EXPECTED_RANKS}"' in text
assert '--gpu-bind=map_gpu:0,1,2,3' in text
assert 'bash "${bundle_dir}/verify_predecessor.sh"' in text
assert '--time=01:55:00' in text
assert 'z4c/telegraph_tau=1.0 z4c/telegraph_kappa=1.0' in text
assert 'z4c/damp_kappa1=0.0 z4c/damp_kappa2=0.0' in text
assert 'z4c/target_kappa1=0.0 z4c/damp_kappa1_max_K=false' in text
assert 'z4c/roll_kappa=false z4c/floor_chi=false' in text
assert 'z4c/diss="${dissipation}" z4c_amr/dchi_max=0.02' in text
assert ('shift_args=(z4c/shift_Gamma=1.0 z4c/shift_eta=2.0\n'
        '        z4c/shift_eta_max_K=false z4c/shift_advect=1.0)') in text
assert ('shift_args=(z4c/shift_Gamma=0 z4c/shift_alpha2Gamma=0 z4c/shift_H=0\n'
        '        z4c/shift_advect=0 z4c/shift_eta=0 z4c/shift_eta_max_K=false)') in text
assert 'assert case["constraint_damping"] is False' in analyzer
assert '"z4c/damp_kappa1": [0.02, 0.0]' in analyzer
assert '"z4c/target_kappa1": [0.02, 0.0]' in analyzer
assert '"z4c/damp_kappa1_max_K": [True, False]' in analyzer
assert '== [0.02, 0.5, 0.5]' in analyzer
assert '"fixed_gamma_driver_eta2",\n        "fixed_gamma_driver_eta2",\n        "zero_shift",' in analyzer
assert 'nargs=3' in analyzer
assert 'analyze_pair.py" pair' not in text
assert 'sub.add_parser("pair")' not in analyzer
source=(root/'source_input.athinput').read_text()
assert re.search(r'^num_levels\s*=\s*9\s*$', source, re.M)
assert re.search(r'^max_ref_lev\s*=\s*8\s*$', source, re.M)
assert re.search(r'^dchi_max\s*=\s*0\.02\s*$', source, re.M)
assert re.search(r'^diss\s*=\s*0\.02\s*$', source, re.M)
assert re.search(r'^shift_eta_max_K\s*=\s*true\s*$', source, re.M)
assert re.search(r'^telegraph_tau\s*=\s*1\.0\s*$', source, re.M)
assert re.search(r'^telegraph_kappa\s*=\s*1\.0\s*$', source, re.M)
print('NOCD_KO_SHIFT_TRIO_STATIC_SELF_TEST_PASS')
PY

if [[ -f "${bundle_dir}/bundle.sha256" ]]; then
  (cd "${bundle_dir}" && sha256sum -c bundle.sha256)
fi

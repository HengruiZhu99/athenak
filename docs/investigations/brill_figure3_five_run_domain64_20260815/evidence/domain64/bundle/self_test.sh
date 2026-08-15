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
allocate=(root/'allocate.sh').read_text()
preflight=(root/'preflight_login.sh').read_text()
verify=(root/'verify_predecessor.sh').read_text()
assert 'cmake ' not in text
assert 'salloc --account="${COLLAPSE_ACCOUNT}"' in allocate
assert '--dependency=' not in allocate
assert '--qos=shared_interactive' in allocate
assert "--constraint='gpu&hbm80g'" in allocate
assert "--constraint='gpu&hbm40g'" not in allocate
assert '--ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1' in allocate
assert 'predecessor-terminal.sha256' in text
assert 'squeue -h -j "${EXPECTED_PREDECESSOR_JOB_ID}"' in preflight
assert 'predecessor-selected.sha256' in verify
assert 'sha256sum -c "${bundle_dir}/predecessor-selected.sha256"' in verify
assert 'sha256sum -c SHA256SUMS' not in verify
assert 'range(3)' in verify
assert 'v1-failure.sha256' in preflight
assert 'allocation/sacct-settled.psv' in preflight
assert 'V3_INCOMPLETE_ROOT' in preflight
assert 'V4_ROOT' in preflight
assert 'V5_ROOT' in preflight
assert 'EXPECTED_V5_EMPTY_PREDECESSOR_LOG_SHA256' in preflight
assert 'V6_ROOT' in preflight
assert 'Cuda memory space failed to allocate 4.883 GiB' in preflight
assert '(cd "${root}" && find run allocation bundle preflight -type f' not in text
assert 'find run allocation bundle preflight -type f' in allocate
assert 'sacct-settled.psv' in allocate
assert text.count('run_case "${CASE_KO05}"') == 1
assert text.count('run_case "${CASE_ZERO_SHIFT_KO05}"') == 1
assert ('run_case "${CASE_KO05}" '
        'brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64_n128 '
        '0.5 fixed_gamma_driver_eta2') in text
assert ('run_case "${CASE_ZERO_SHIFT_KO05}" '
        'brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_n128 '
        '0.5 zero_shift') in text
assert 'mesh_refinement/num_levels="${EXPECTED_NUM_LEVELS}"' in text
assert 'z4c_amr/max_ref_lev="${EXPECTED_MAX_REF_LEV}"' in text
assert 'time/nmb_total_limit="${EXPECTED_NMB_TOTAL_LIMIT}"' in text
assert 'mesh_refinement/max_nmb_per_rank="${EXPECTED_MAX_NMB_PER_RANK}"' in text
assert 'z4c/shift_eta_max_K=false' in text
assert '--ntasks="${EXPECTED_RANKS}"' in text
assert '--gpu-bind=map_gpu:0' in text
assert 'node-profile.txt' in text
assert "grep -Eq '(^|,)hbm80g(,|$)'" in text
assert 'EXPECTED_GPU_NAME//_/ ' in text
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
assert 'NVIDIA A100-SXM4-80GB' in analyzer
assert '"mesh/x1max": [16.0, 64.0]' in analyzer
assert '"mesh/x2min": [-16.0, -64.0]' in analyzer
assert '"mesh/x2max": [16.0, 64.0]' in analyzer
assert '"base_dx": [0.25, 0.25]' in analyzer
assert '== [0.5, 0.5]' in analyzer
assert '"fixed_gamma_driver_eta2",\n        "zero_shift",' in analyzer
assert 'nargs=2' in analyzer
assert 'analyze_pair.py" pair' not in text
assert 'sub.add_parser("pair")' not in analyzer
source=(root/'source_input.athinput').read_text()
ledger=(root/'predecessor-selected.sha256').read_text().splitlines()
assert len(ledger) == 42
assert all('/bin/' not in line and '/rst/' not in line for line in ledger)
assert any(line.endswith('  allocation/sacct-settled.psv') for line in ledger)
assert any(line.endswith('  run/comparison.json') for line in ledger)
assert sum('/bindings/rank_binding_' in line for line in ledger) == 12
assert sum(line.endswith('.z4c.user.hst') for line in ledger) == 3
assert sum(line.endswith('/run.log') for line in ledger) == 3
assert re.search(r'^num_levels\s*=\s*9\s*$', source, re.M)
assert re.search(r'^max_ref_lev\s*=\s*8\s*$', source, re.M)
assert re.search(r'^dchi_max\s*=\s*0\.02\s*$', source, re.M)
assert re.search(r'^diss\s*=\s*0\.02\s*$', source, re.M)
assert re.search(r'^max_nmb_per_rank\s*=\s*16384\s*$', source, re.M)
assert re.search(r'^nmb_total_limit\s*=\s*16384\s*$', source, re.M)
assert re.search(r'^nx1\s*=\s*256\s*$', source, re.M)
assert re.search(r'^x1min\s*=\s*0\.0\s*$', source, re.M)
assert re.search(r'^x1max\s*=\s*64\.0\s*$', source, re.M)
assert re.search(r'^nx2\s*=\s*512\s*$', source, re.M)
assert re.search(r'^x2min\s*=\s*-64\.0\s*$', source, re.M)
assert re.search(r'^x2max\s*=\s*64\.0\s*$', source, re.M)
assert re.search(r'^shift_eta_max_K\s*=\s*true\s*$', source, re.M)
assert re.search(r'^telegraph_tau\s*=\s*1\.0\s*$', source, re.M)
assert re.search(r'^telegraph_kappa\s*=\s*1\.0\s*$', source, re.M)
print('DOMAIN64_NOCD_KO05_SHIFT_PAIR_STATIC_SELF_TEST_PASS')
PY

if [[ -f "${bundle_dir}/bundle.sha256" ]]; then
  (cd "${bundle_dir}" && sha256sum -c bundle.sha256)
fi

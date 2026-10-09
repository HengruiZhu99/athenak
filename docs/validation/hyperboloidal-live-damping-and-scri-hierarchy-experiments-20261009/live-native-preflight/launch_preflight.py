"""Fresh finite-Omega native live C0 damping-profile reference/short preflights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PYTHON = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
INDEX = 'fcfd4a740fc25e999d0417598015608c009bf61de032aff5266f1f843e2d6b59'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
stage = sys.argv[1]
assert stage in {'reference', 'short'}
launch_path = HERE/(stage+'-launch.json')
assert not launch_path.exists()
base_path = FAMILY/'native-build-receipt.json'
base = json.loads(base_path.read_text())
build_path = HERE/'native-build/build-receipt.json'
build = json.loads(build_path.read_text())
assert sha(build_path) == '5ade5d3c80b68913301957a7093929a6793af5da8a9fc19c4215f140d7e49a72'
assert build['base_build_receipt_sha256'] == sha(base_path)
assert build['live_gate_index_sha256'] == INDEX
assert build['link_exit_status'] == 0
assert build['base_source_overlay_and_link_inputs_unchanged'] is True
for key, value in base['source_sha256'].items():
    assert sha(ROOT/key) == value, key
for key, value in base['overlay_sha256'].items():
    assert sha(FAMILY/key) == value, key
for catalog in ['private_source_sha256', 'all_compiled_repository_dependencies_sha256']:
    for key, value in build[catalog].items():
        assert sha(ROOT/key) == value, key
for key, value in build['reused_base_link_inputs_sha256'].items():
    assert sha(ROOT/key) == value['sha256'], key
for index, row in enumerate(build['compile_results']):
    assert row['exit_status'] == 0
    assert sha(HERE/f'native-build/object-{index}.o') == row['private_object_sha256']
    assert sha(HERE/f'native-build/object-{index}.d') == row['dependency_file_sha256']
assert sha(HERE/'build_live_native.py') == build['script_sha256']
assert (HERE/'native-build/build-source.py').read_bytes() == (HERE/'build_live_native.py').read_bytes()
assert sha(HERE/'native-build/include/live_damping_profile.hpp') == '69bbbc137486eb3398f94ed50a8b04583c375da049372b82b19330d8432fc153'
index_path=ROOT/build['live_gate_index']
assert sha(index_path)==INDEX
gate=json.loads(index_path.read_text())
for key,value in gate['files'].items():
    path=index_path.parent/key
    wanted=value if isinstance(value,str) else value['sha256']
    assert sha(path)==wanted
    if isinstance(value,dict) and 'bytes' in value:
        assert path.stat().st_size==value['bytes']
receipt_path=index_path.parent/'receipt.json'
assert sha(receipt_path)=='e038769458c15e7cd1c5a956ef75bd4508b777c981d7ac7da9594c273dcdf229'
receipt=json.loads(receipt_path.read_text())
assert receipt['passed_finite_Omega_local_numerical_gate'] is True
assert receipt['global_native_or_scri_stability_accepted'] is False
assert receipt['sources_unchanged'] is True and len(receipt['source_before'])==384
assert all(x['returncode']==0 for x in receipt['commands'])
review_path=ROOT/'build-layer-research/continuum/independent-live-damping-review/immutable-independent-live-review-20261009/index.json'
assert sha(review_path)=='54e403b71a9e0398fe7700cf2ab7dad836167019463e8783831563e4a28ab54d'
review=json.loads(review_path.read_text())
for name,spec in review['files'].items():
    wanted=spec if isinstance(spec,str) else spec['sha256']
    assert sha(review_path.parent/name)==wanted
override = HERE/'native.athinput'
assert override.read_bytes()==(FAMILY/'native.athinput').read_bytes()
exe=ROOT/build['executable']
assert sha(exe)==build['executable_sha256']=='580b6043f906ea37ae4870db2cd63d68f344c634c28319d88dafe53308d56005'
command=[PYTHON,str(ROOT/'tst/hyperboloidal/run_layer_validation.py'),str(exe),
    str(HERE/stage),'--suite','reference-long' if stage=='reference' else 'long',
    '--overrides',str(override),'--duration','.05' if stage=='reference' else '.02',
    '--output-cadence','.025' if stage=='reference' else '.01']
record={
 'scope':'Fixed finite-Omega live C0 damping-profile exploratory native preflight; no stable pulse or scri closure.',
 'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'compiled_production_implementation':build['compiled_implementation'],
 'native_build_receipt_sha256':sha(build_path),'native_executable_sha256':sha(exe),
 'launch_script_sha256':sha(Path(__file__)),'override_sha256':sha(override),
 'all_recorded_source_private_object_dependency_link_hashes_verified':True,
 'independent_review_index_sha256':sha(review_path),
 'live_gate_index_sha256':INDEX,'live_gate_receipt_sha256':sha(receipt_path),
 'shell_pole_diagnostic':'Actual C0 ConformalRHS pole using same live kappa2 helper in live/reference assembly.',
 'command':command,'stage':stage}
if stage=='short':
    reference_audit=HERE/'audit/reference.json'
    accepted=json.loads(reference_audit.read_text())
    assert accepted['status']=='PASS'
    record['reference_audit_sha256']=sha(reference_audit)
launch_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
subprocess.run(command,cwd=ROOT,check=True)

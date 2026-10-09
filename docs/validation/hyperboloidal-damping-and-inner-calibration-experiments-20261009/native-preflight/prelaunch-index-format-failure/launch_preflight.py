"""Fresh finite-Omega native C0 damping-profile reference/short preflights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PYTHON = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
INDEX = 'c9180b5bbedb2a0069a54853768a96f8b0f69736c30a8bc9c277b77f11fc39ea'
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
assert sha(build_path) == '062ba8b3103f90233354134ab61a99892eb5cd8a7b61de5cf0d396dc61129cfe'
assert build['base_build_receipt_sha256'] == sha(base_path)
assert build['profile_gate_index_sha256'] == INDEX
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
assert sha(HERE/'build_profiled_native.py') == build['script_sha256']
assert (HERE/'native-build/build-source.py').read_bytes() == (HERE/'build_profiled_native.py').read_bytes()
assert sha(HERE/'native-build/include/damping_profile.hpp') == '64ba382f509e81fe347b96e915d3933ee7188c97fd6a054524b1aa187a891532'
index_path=ROOT/build['profile_gate_index']
assert sha(index_path)==INDEX
gate=json.loads(index_path.read_text())
for key,value in gate['files'].items():
    path=index_path.parent/key
    assert sha(path)==value['sha256'] and path.stat().st_size==value['bytes']
receipt_path=index_path.parent/'receipt.json'
assert sha(receipt_path)=='d832017b2091f9f9e618fd9f4b6cfdbce0afcb19972ad43eb8217bd4681bdbaa'
receipt=json.loads(receipt_path.read_text())
assert receipt['passed_profile_local_gates'] is True
assert receipt['global_native_or_scri_stability_accepted'] is False
assert receipt['sources_unchanged'] is True and len(receipt['source_before'])==378
assert all(x['returncode']==0 for x in receipt['commands'])
override = HERE/'native.athinput'
assert override.read_bytes()==(FAMILY/'native.athinput').read_bytes()
exe=ROOT/build['executable']
assert sha(exe)==build['executable_sha256']=='b7e82aea627cd10b89958c993c1429b0322df60bcb62dd1cd7ce46b2a5341b84'
command=[PYTHON,str(ROOT/'tst/hyperboloidal/run_layer_validation.py'),str(exe),
    str(HERE/stage),'--suite','reference-long' if stage=='reference' else 'long',
    '--overrides',str(override),'--duration','.05' if stage=='reference' else '.02',
    '--output-cadence','.025' if stage=='reference' else '.01']
record={
 'scope':'Fixed finite-Omega C0 damping-profile exploratory native preflight; no stable pulse or scri closure.',
 'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'compiled_production_implementation':build['compiled_implementation'],
 'native_build_receipt_sha256':sha(build_path),'native_executable_sha256':sha(exe),
 'launch_script_sha256':sha(Path(__file__)),'override_sha256':sha(override),
 'all_recorded_source_private_object_dependency_link_hashes_verified':True,
 'profile_gate_index_sha256':INDEX,'profile_gate_receipt_sha256':sha(receipt_path),
 'shell_pole_diagnostic':'Actual C0 ConformalRHS pole using same prescribed kappa2 profile in live/reference assembly.',
 'command':command,'stage':stage}
if stage=='short':
    reference_audit=HERE/'audit/reference.json'
    accepted=json.loads(reference_audit.read_text())
    assert accepted['status']=='PASS'
    record['reference_audit_sha256']=sha(reference_audit)
launch_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
subprocess.run(command,cwd=ROOT,check=True)

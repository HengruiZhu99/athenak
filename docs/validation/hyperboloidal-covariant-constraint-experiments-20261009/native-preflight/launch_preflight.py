"""Launch one fresh, bounded C1 native preflight with exact build/gate identity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PYTHON = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
INDEX_V1 = '321439833808976b1b6ebfc99e443b0f61e0018b4a0aea257986ae55f0e846a8'
INDEX_V2 = 'd8d137e4422dec83ec685f0fee45fc80d3363cbf6a3252fec8b8d41c53c485ed'


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
assert sha(build_path) == 'd3ce419ae6406570ee95ed9e206305e5e13788f44c461a1897db907b11e54e19'
assert build['base_build_receipt_sha256'] == sha(base_path)
assert build['stiffness_gate_index_sha256'] == INDEX_V1
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
assert sha(HERE/'build_covariant_native.py') == build['script_sha256']
assert (HERE/'native-build/build-source.py').read_bytes() == (HERE/'build_covariant_native.py').read_bytes()
gate_root = ROOT/'build-layer-research/continuum/covariant-z4-candidate'
gates = {}
for name, expected in [('immutable-C1-stiffness-20261009', INDEX_V1),
                       ('immutable-C1-stiffness-v2-20261009', INDEX_V2)]:
    index_path = gate_root/name/'index.json'
    assert sha(index_path) == expected
    gate = json.loads(index_path.read_text())
    for key, value in gate['files'].items():
        path = index_path.parent/key
        assert sha(path) == value['sha256'] and path.stat().st_size == value['bytes']
    receipt_path = index_path.parent/'receipt.json'
    assert sha(receipt_path) == 'efd53d5e40ba0437c2c048d9334abc68aceb14122e52fc0a11366acc52362bfa'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['passed_finite_Omega_local_numerical_gate'] is True
    assert receipt['native_or_scri_stability_accepted'] is False
    assert len(receipt['source_before']) == 370
    gates[name] = {'index_sha256': expected, 'receipt_sha256': sha(receipt_path)}
override = HERE/'native.athinput'
assert override.read_bytes() == (FAMILY/'native.athinput').read_bytes()
exe = ROOT/build['executable']
assert sha(exe) == build['executable_sha256'] == '84bb958271395e019c1f26d56f50253f64b7e87ade1bfd40d928d443d03918b6'
command = [PYTHON, str(ROOT/'tst/hyperboloidal/run_layer_validation.py'), str(exe),
           str(HERE/stage), '--suite', 'reference-long' if stage == 'reference' else 'long',
           '--overrides', str(override), '--duration', '.05' if stage == 'reference' else '.02',
           '--output-cadence', '.025' if stage == 'reference' else '.01']
record = {
    'scope': 'Bounded strict-interior repaired-C1 native preflight only; no scri/pulse acceptance.',
    'launch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'compiled_production_implementation': build['compiled_implementation'],
    'native_build_receipt_sha256': sha(build_path), 'native_executable_sha256': sha(exe),
    'launch_script_sha256': sha(Path(__file__)), 'override_sha256': sha(override),
    'all_recorded_source_private_object_dependency_link_hashes_verified': True,
    'gates': gates, 'gate_v2_erratum': 'Corrected source count and explicit kappa5 leading pole; '
                                    'numeric receipt and kappa10 candidate unchanged. As-built v1 pin retained.',
    'command': command, 'stage': stage,
}
if stage == 'short':
    reference_audit = HERE/'audit/reference.json'
    accepted = json.loads(reference_audit.read_text())
    assert accepted['status'] == 'PASS'
    record['reference_audit_sha256'] = sha(reference_audit)
launch_path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
subprocess.run(command, cwd=ROOT, check=True)

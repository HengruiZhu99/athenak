"""Read-only hash, receipt and stated-scope verification of the frozen gate."""
from pathlib import Path
import argparse
import hashlib
import json

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--sha256', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
gate = args.index.resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(args.index) == args.sha256
index = json.loads(args.index.read_text())
for name, entry in index['files'].items():
    expected = entry if isinstance(entry, str) else entry['sha256']
    assert sha(gate/name) == expected, name
assert len(index['files']) == index['file_count'] == 73
assert sum((gate/name).stat().st_size for name in index['files']) == index['total_bytes'] == 3342320
receipt = json.loads((gate/'receipt.json').read_text())
assert sha(gate/'receipt.json') == index['receipt_sha256']
assert receipt['sources_unchanged'] and receipt['source_before'] == receipt['source_after']
assert len(receipt['source_after']) == index['input_count'] == 382
for name, expected in receipt['source_after'].items():
    assert sha(root/name) == expected, name
assert len(receipt['commands']) == index['command_count'] == 14
for command in receipt['commands']:
    assert command['returncode'] == 0
    for stream in ('stdout', 'stderr'):
        assert sha(gate/command[stream]) == command[stream+'_sha256']
    assert (gate/command['stderr']).stat().st_size == 0
assert receipt['release_debug_json_equal']
assert (gate/'full20.json').read_bytes() == (gate/'full20-debug.json').read_bytes()
assert (gate/'nonlinear.json').read_bytes() == (gate/'nonlinear-debug.json').read_bytes()
large = index['large_outputs_outside_snapshot']
for entry in large.values():
    original = root/entry['original_repo_path']
    assert sha(original) == entry['sha256']
    assert original.stat().st_size == entry['bytes']
prior = json.loads((gate/'prior-controls.json').read_text())
for name, expected in prior['files'].items():
    assert sha(root/name) == expected, name
assert index['helper_sha256'] == sha(gate/'q_null_feedback.hpp')
assert index['factored_base_sha256'] == sha(gate/'factored_base.hpp')
report = json.loads((gate/'check-report.json').read_text())
principal = json.loads((gate/'check-principal.stdout').read_text())
assert report['passed_local_pole_source_principal_corner_gates']
assert not report['native_or_global_accepted']
assert principal['passed_kernel_cases'] == 504
assert principal['max_kernel_symbol_error'] < 1e-12
assert principal['harmonic_endpoint_complete']
assert report['Einstein_corner_limit_error'] < 1e-9
assert report['prior_physical_projection_corner_FD_error'] < 2e-7
assert report['nonlinear']['tiny_noncore_rows'] == 320
assert report['nonlinear']['reference_and_blend_rows'] == 640
assert report['nonlinear']['source_rows'] == 160
assert report['nonlinear']['tiny_positive_core_rows'] == 48
text = (gate/'nonlinear.cpp').read_text()
assert 'for(double xi:{1.5,1./a})' in text
assert 'g.scri_lapse_damping=xi' in text
helper = (gate/'q_null_feedback.hpp').read_text()
assert 'const T delta=WeightedNullDifference(p,u);' in helper
assert 'f.beta[i]+=q.pole.beta[i]/O' in helper
assert 'auto out=par.physical_inner&&W==T(0)?' in helper
assert 'scri_lapse_damping' not in helper  # Inherited, never overridden.
documentation = (gate/'DERIVATION.md').read_text()
assert 'There are no transition\nfinite-frequency matrices' in documentation
assert 'No old preferred Box identity is claimed there' in documentation
out = {'status': 'PASS_INDEPENDENT_FROZEN_LOCAL_GATE_REVIEW',
       'gate_index': str(args.index.resolve()), 'gate_index_sha256': args.sha256,
       'gate_file_count': 73, 'gate_bytes': 3342320, 'input_paths_rechecked': 382,
       'commands_rechecked': 14, 'prior_control_files_rechecked': len(prior['files']),
       'large_binary_records_rechecked': len(large),
       'helper_sha256': index['helper_sha256'],
       'factored_base_sha256': index['factored_base_sha256'],
       'receipt_sha256': index['receipt_sha256'],
       'principal_cases': 504, 'release_ASan_UBSan_output_bytes_equal': True,
       'inherited_xi_transition_nonlin_choices': ['1.5', '1/a'],
       'scientific_correction_required': False,
       'external_prose_clarification': 'DERIVATION shift weighted log identity must read alpha^2 dlog(alpha/h), rather than alpha^2 dlog(alpha); actual alpha2du implementation is correct.',
       'scope': 'Reference outer pole, nonlinear finite-Omega source/factoring/transition identities and complete principal only; finite-Fourier transition, R0/Taylor closure, global/native/BH and uniform lapse claims remain unaccepted.'}
args.output.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))

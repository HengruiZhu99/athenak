"""Read-only frame/cubic source, metadata and saved-matrix review receipt.

No actual-kernel execution or exact rowspace reconstruction is repeated.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


gates = [
    ('frame', 'build-layer-research/continuum/q-null-boundary-frame/'
     'immutable-Q-boundary-frame-timejet-20261009',
     '79cf6ec57336b75e95e74e71287273ef9d6f18606733432229278eb42bcc7408'),
    ('cubic', 'build-layer-research/continuum/q-null-curvature-ideal/'
     'immutable-Q-Einstein-cubic-null-curvature-timejet-20261009',
     '0302a9da72f1a36f5fd6bf7dc9311fccc08c97ac1eace3282423a2ee38517f13')]
checked = []
for name, directory, pin in gates:
    base = ROOT/directory
    assert sha(base/'index.json') == pin
    index = json.loads((base/'index.json').read_text())
    for row in index['files']:
        path = base/row['path']
        assert sha(path) == row['sha256']
        if 'bytes' in row:
            assert path.stat().st_size == row['bytes']
    receipt = json.loads((base/'receipt.json').read_text())
    assert all(row['returncode'] == 0 for row in receipt['commands'])
    assert all((base/row['stderr']).stat().st_size == 0
               for row in receipt['commands'])
    assert receipt['sources_unchanged']
    assert receipt['source_before'] == receipt['source_after']
    checked.append({
        'name': name, 'index': directory+'/index.json', 'sha256': pin,
        'indexed_files_verified': len(index['files']),
        'source_input_count': len(receipt['source_before']),
        'passing_commands_empty_stderr': len(receipt['commands']),
        'saved_commands_seconds': sum(q['seconds'] for q in receipt['commands'])})

base = ROOT/gates[1][1]
assert ((base/'actual-release.json').read_bytes()
        == (base/'actual-debug.json').read_bytes())
actual = json.loads((base/'actual-release.json').read_text())
assert actual['labels'] and len(actual['labels']) == 246
maximum = 0.
columns = 0
for radius in actual['radii']:
    a = radius['a']
    labels = {key: i for i, key in enumerate(actual['labels'])}
    terms = {
        'H[0,0,0]': 1/6, 'H[1,0,0]': -2/a, 'H[2,0,0]': 1/a**2,
        'H[0,2,0]': 1/3, 'H[0,0,2]': 1/3,
        'M0[0,0,0]': -2/a, 'M0[1,0,0]': 2/a**2,
        'M1[0,1,0]': -1/a, 'M2[0,0,1]': -1/a,
        'Z0[0,0,0]': -12/a**2, 'Z0[1,0,0]': 4/a**3,
        'Theta[0,0,0]': 4/a, 'Theta[1,0,0]': -12/a**2,
        'Theta[2,0,0]': 4/a**3, 'N[0,0,0]': 1, 'N[1,0,0]': 2/a}
    assert len(radius['orientations']) == 2
    for orientation in radius['orientations']:
        assert len(orientation['columns']) == 400
        for column in orientation['columns']:
            lhs = column['N1_t']+column['delta_Rq']/a**2
            rhs = sum(factor*column['E'][labels[key]]
                      for key, factor in terms.items())
            maximum = max(maximum, abs(lhs-rhs))
            columns += 1
assert columns == 3200 and maximum < 1e-12

reviewed = [HERE/'audit-draft.md', HERE/'archive-README.md', Path(__file__),
            ROOT/'build-layer-research/continuum/q-null-curvature-ideal/'
            'NOTATION-ADDENDUM.md']
result = {
    'passed_read_only_source_math_saved_result_and_draft_review': True,
    'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=ROOT, text=True).strip(),
    'frozen_inputs': checked,
    'reviewed_file_pins': {str(p.relative_to(ROOT)): sha(p) for p in reviewed},
    'saved_matrix_readback': {
        'columns': columns,
        'common_fixed_frame_sparse_identity_max_absolute_error': maximum,
        'scope': 'Cheap numerical readback of the explicit saved identity only; '
        'not an independent exact rowspace/kernel rerun.'},
    'math_checks': [
        'Moving spatial normal in Y=beta+alpha*s and qtt Lie drift are retained; '
        'fixed coordinate frame and intrinsic roundness are distinct.',
        'Cubic chart spans400 primitive tangent Taylor columns; all246 conditions '
        'respect the available derivative order and fixed Cartesian M/Z frame.',
        'L=(Nregular)_1+(Npole)_2 consumes only second state jets; pure cubic '
        'zeros, next-pole value/first jets and curvature-rate order are checked '
        'explicitly in the inspected source.',
        'Independent induced-curvature formula retains tangent-frame/graph '
        'derivatives and the -2trace cut-metric term.',
        'Rank127/nullity273 and augmented rank128 are exact reconstructed-matrix '
        'statements, distinguished from raw floating kernels and Einstein germs.',
        'Draft counts, errors, command times, oracle parameter scopes and '
        'conditional first-time tangency match the frozen reports.'],
    'scientific_corrections': [],
    'notation_clarification': 'R0 denotes both the leading pole map elsewhere '
    'and the regular Taylor coefficient in F0=R0+S1. Prefer '
    'F0=regular_0+pole_1,F1=regular_1+pole_2 in the readable draft.',
    'interpretation': 'The boundary metric transport formula is conditional '
    'on smooth conformal Einstein/null compatibility, not ADM initial '
    'constraints alone. Necessary finite jets are not full Einstein germs. '
    'No full ideal/hierarchy invariance, nonlinear closure, radiative-data '
    'admission, amplitude instability, boundary condition or evolution result.',
    'new_scientific_execution': False,
    'draft_or_frozen_mutation': False}
(HERE/'independent-draft-review.json').write_text(
    json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'passed': True, 'receipt_sha256':
                  sha(HERE/'independent-draft-review.json')}, indent=2))

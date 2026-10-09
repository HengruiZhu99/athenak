"""Freeze the reviewed advisory bytes; does not rerun either algebra checker."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
DEST = P/'immutable-full-ball-radial-advisory-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not DEST.exists()
receipt = json.loads((P/'receipt.json').read_text())
assert receipt['passed_advisory_checks']
for name, pin in receipt['source_hashes_before_equals_after'].items():
    assert sha(ROOT/name) == pin
for name, pin in receipt['report_hashes'].items():
    assert sha(P/name) == pin
assert all(c['returncode'] == 0 for c in receipt['commands'])
root_review = {
    'kind': 'root-read-only-advisory-review-record-from-parent-message',
    'reviewer': '/root', 'required_corrections': [],
    'verbatim_parent_message': (
        'Root read ASSESSMENT/FINITE-RADIUS/VARIATIONAL-SAT and both check '
        'sources; no corrections. Checked mass degree bound '
        'M>=N+floor(L/2), envelope physical boundary terms, A² involution '
        'symmetrizer, momentum-only Gram/Schur and adjoint energy signs, and '
        'unbounded trace examples. You may include root read-only review in '
        'advisory freeze (no new kernel/model run). Please freeze after '
        'draft review.'),
    'reviewed_source_hashes': receipt['source_hashes_before_equals_after'],
    'scope': 'Advisory only; no radial PDE operator or solver admission.'}
(P/'root-review.json').write_text(json.dumps(root_review, indent=2)+'\n')
DEST.mkdir()
files = [p for p in P.rglob('*') if p.is_file() and DEST not in p.parents
         and '__pycache__' not in p.parts and p.suffix != '.pyc']
for p in files:
    assert p.stat().st_size <= 1024*1024
    q = DEST/p.relative_to(P)
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)

identity = json.loads((P/'identity-report.json').read_text())
principal = json.loads((P/'boundary-principal-report.json').read_text())
sources = dict(principal['source_pins'])
sources['src/z4c/hyperboloidal/radial_sbp.hpp'] = (
    identity['source_pins']['scalar_radial_SBP'])
basis = ('build-layer-research/continuum/total-j-harmonic-basis/'
         'immutable-Cartesian-total-J-basis-20261009')
sources[basis+'/index.json'] = identity['frozen_basis_index_sha256']
sources[basis+'/total_j_basis.py'] = identity['source_pins']['basis_source']
inputs = []
for name, pin in sorted(sources.items()):
    p = ROOT/name
    assert sha(p) == pin
    entry = {'original_path': name, 'sha256': pin, 'bytes': p.stat().st_size}
    if p.stat().st_size <= 1024*1024:
        q = DEST/'inputs'/name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        entry['captured_path'] = str(q.relative_to(DEST))
    else:
        entry['metadata_only'] = True
    inputs.append(entry)
(DEST/'input-metadata.json').write_text(json.dumps(inputs, indent=2)+'\n')
entries = []
finite_json = 0
for p in sorted(DEST.rglob('*')):
    if not p.is_file():
        continue
    if p.suffix == '.json':
        json.loads(p.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError('nonfinite JSON '+x)))
        finite_json += 1
    entries.append({'path': str(p.relative_to(DEST)), 'sha256': sha(p),
                    'bytes': p.stat().st_size})
index = {
    'kind': 'immutable-math-only-full-ball-radial-and-finite-boundary-advisory',
    'freeze_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=ROOT, text=True).strip(),
    'scope': ('Regular-origin representation, quadrature/weak identities, '
              'actual retained harmonic principal and frozen/variational '
              'SAT energy-work algebra only. No radial PDE operator, new '
              'kernel/model run, eigenproblem, evolution or solver admission.'),
    'file_count': len(entries), 'bytes': sum(f['bytes'] for f in entries),
    'finite_json_count': finite_json,
    'checks': len(receipt['commands']),
    'check_seconds': receipt['seconds'],
    'historical_structural_assertion_failure_preserved': True,
    'root_read_only_review_included': True,
    'large_inputs_metadata_only': [f for f in inputs if f.get('metadata_only')],
    'files': entries}
(DEST/'index.json').write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
for f in entries:
    p = DEST/f['path']
    assert sha(p) == f['sha256'] and p.stat().st_size == f['bytes']
print(json.dumps({'index': str((DEST/'index.json').relative_to(ROOT)),
                  'sha256': sha(DEST/'index.json'), 'file_count': len(entries),
                  'bytes': index['bytes'], 'finite_json_count': finite_json,
                  'large_input_records': len(index['large_inputs_metadata_only'])},
                 indent=2))

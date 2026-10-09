"""Collect completed factored Q/null source, native and rejected Cartesian evidence once."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-Q-early-and-jet-experiments-20261009'
FILES = {}
OMITTED = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def add(source, target):
    data = source.read_bytes()
    if source.suffix in {'.npz', '.npy', '.o', '.a', '.bin', '.rst', '.raw'} or len(data)>1048576:
        OMITTED[target] = {'source': str(source.relative_to(ROOT)), 'sha256': sha(data), 'bytes': len(data)}
        return
    if source.suffix == '.json':
        finite(json.loads(data))
    assert target not in FILES
    output = DEST/target
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    assert output.read_bytes() == data
    FILES[target] = {'source': str(source.relative_to(ROOT)),
                     'sha256': sha(data), 'bytes': len(data)}


def frozen(folder, index, expected, target):
    base = ROOT/folder
    path = base/index
    assert sha(path.read_bytes()) == expected, path
    record = json.loads(path.read_text())
    entries = record.get('files', record.get('small_files'))
    assert entries
    for name, spec in (entries.items() if isinstance(entries,dict) else ((entry['path'],entry) for entry in entries)):
        source = ROOT/name if name.startswith('build-layer-research/') else base/name
        data = source.read_bytes()
        wanted = spec if isinstance(spec, str) else spec['sha256']
        assert sha(data) == wanted, source
        if isinstance(spec, dict) and 'bytes' in spec:
            assert len(data) == spec['bytes'], source
        relative = source.relative_to(base) if source.is_relative_to(base) else source.relative_to(ROOT)
        add(source, target+'/'+str(relative))
    add(path, target+'/'+index)


assert not DEST.exists(), 'Refuse to mutate an existing archive'
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == '316c02576ba0b30643eeb762fdfcc88fe96d6668'
gates = [('continuum/q-null-early-feedback/immutable-Q-null-early-feedback-local-20261009', 'index.json', '902d875e6da80bf3b10e7103aa59cf123c4d8a034862f0a5d9be6f9e220c56c0', 'early-local'), ('continuum/independent-q-early-review/immutable-independent-Q-early-support-review-20261009', 'index.json', '273fdab75ae847d4a54a6cd8b50334a215a839969de39c5bb7378b50fa24b19a', 'early-review'), ('boundary/full-tensor-conformal-q-early-weight/immutable-Q-early-weight-global-screen-20261009', 'index.json', '296c1f8dc21915d71331bbe3436ac1f0f797afa66614a17aba109a0df1052432', 'early-global-negative'), ('continuum/q-null-jet-followup/immutable-Q-null-firstjet-map-20261009', 'index.json', '5490057dfc04e060bca65ec7ee1a3bb363a34cab0fe5c890f23d477a7bda7ec1', 'firstjet-map'), ('continuum/independent-q-jet-review/immutable-independent-Q-firstjet-review-20261009', 'index.json', '32a4e257413ee50a8b9d09c84baa2898032d6d866a8c0247bd9967f3780578e9', 'firstjet-review'), ('continuum/q-null-invariant-jets/immutable-Q-sigma5-Einstein-null-noninvariance-20261009', 'index.json', '4cfbc8ed743c46787f617c33fdbe9875ec092fd117e82f1daddf5e165f2ef29a', 'Einstein-null-noninvariance'), ('q-null-independent-box-derivation/immutable-independent-ADM-Box-null-identity-20261009', 'index.json', '78c6870117cf1fb74e1175cda6abc38b08b5ac9092f3fce7d904bf8dd958e585', 'independent-ADM-Box'), ('continuum/independent-q-invariant-review/immutable-independent-Einstein-Q-null-invariance-review-20261009', 'index.json', 'aafc5a2aa870d3185f3b474f123cae16cb5240ddbd384dc28ffdd8c8d534c259', 'invariant-independent-review')]
assert all('PENDING' not in item for row in gates for item in row)
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_q_early_jet_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Earlier null-feedback weight source and negative Cartesian control, actual full Q first-jet residue map and independent review, initial Einstein sigma-five quadratic-null time-jet obstruction, independent ADM/Box identity. No invariant nonlinear closure, stable pulse, production promotion or BH admission.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES, 'omitted_large_payloads': OMITTED,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

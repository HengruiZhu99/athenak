"""Collect completed conditional null/curvature finite jets once after full preflight."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-Q-intrinsic-curvature-jet-experiments-20261009'
FILES = {}
OMITTED = {}
PENDING_BYTES = {}


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


def queue(source, target):
    data = source.read_bytes()
    assert target not in FILES and target not in OMITTED, target
    assert not Path(target).is_absolute() and '..' not in Path(target).parts
    spec = {'source': str(source.relative_to(ROOT)),
            'sha256': sha(data), 'bytes': len(data)}
    if source.suffix == '.json':
        finite(json.loads(data))
    if source.suffix in {'.npz', '.npy', '.o', '.a', '.bin', '.rst', '.raw'} or len(data)>1048576:
        OMITTED[target] = spec
        return
    FILES[target] = spec
    PENDING_BYTES[target] = data


def frozen(folder, expected, target):
    base = ROOT/'build-layer-research'/folder
    path = base/'index.json'
    assert sha(path.read_bytes()) == expected, path
    record = json.loads(path.read_text())
    finite(record)
    entries = record.get('files', record.get('small_files'))
    assert entries
    for name, spec in (entries.items() if isinstance(entries, dict) else ((entry['path'], entry) for entry in entries)):
        source = ROOT/name if name.startswith('build-layer-research/') else base/name
        data = source.read_bytes()
        wanted = spec if isinstance(spec, str) else spec['sha256']
        assert sha(data) == wanted, source
        if isinstance(spec, dict) and 'bytes' in spec:
            assert len(data) == spec['bytes'], source
        relative = source.relative_to(base) if source.is_relative_to(base) else source.relative_to(ROOT)
        queue(source, target+'/'+str(relative))
    queue(path, target+'/index.json')


assert not DEST.exists(), 'Refuse to mutate an existing archive'
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == 'a7260897b92d12d5fc1ceb2f3dc6a886676a663f'
assert (HERE/'archive-README.md').is_file(), 'Missing README preflight'
assert (HERE/'independent-draft-review.json').is_file(), 'Missing independent review preflight'
assert (HERE/'independent-tensor-review.json').is_file(), 'Missing tensor review preflight'
frozen('continuum/q-null-boundary-frame/immutable-Q-boundary-frame-timejet-20261009',
       '79cf6ec57336b75e95e74e71287273ef9d6f18606733432229278eb42bcc7408', 'frame')
frozen('continuum/q-null-curvature-ideal/immutable-Q-Einstein-cubic-null-curvature-timejet-20261009',
       '0302a9da72f1a36f5fd6bf7dc9311fccc08c97ac1eace3282423a2ee38517f13', 'cubic')
frozen('continuum/q-curvature-tensor-freedoms/immutable-Q-curvature-tensor-freedoms-20261009',
       'c16dbe6eb947a9c5f8710c297c042868c3dd74581742df9cf81786875316380d', 'tensor-freedoms')
addendum = ROOT/'build-layer-research/continuum/q-null-curvature-ideal/NOTATION-ADDENDUM.md'
assert sha(addendum.read_bytes()) == '0c0386bc0175914f84788167f584f1618d011baf96a9558826181abc16a82aff'
queue(addendum, 'NOTATION-ADDENDUM.md')
for name in ('root-review.json', 'independent-draft-review.json', 'root-tensor-review.json', 'independent-tensor-review.json'):
    queue(HERE/name, 'reviews/'+name)
queue(HERE/'review_frozen_evidence.py', 'reviews/review_frozen_evidence.py')
queue(HERE/'review_tensor_evidence.py', 'reviews/review_tensor_evidence.py')
queue(HERE/'audit-draft.md', 'reviews/reviewed-draft.md')
queue(HERE/'archive-README.md', 'README.md')
queue(HERE/'archive-tensor-addendum.md', 'TENSOR-ADDENDUM.md')
queue(Path(__file__), 'collect_q_curvature_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
      'verify_archive.py')
catalog = {'scope': 'Actual moving-normal boundary-frame calculation and complete cubic finite Einstein/null/shear chart. Conditional linear first-time null/intrinsic-curvature tangency at four reference scales. No complete hierarchy, nonlinear exact-scri assembly, radiative-data admission or evolution/stability result.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES, 'omitted_large_payloads': OMITTED,
           'large_artifacts': 'Metadata/hashes only in original copied indexes and receipts.'}
finite(catalog)
catalog_bytes = (json.dumps(catalog, indent=2, allow_nan=False)+'\n').encode()
# No output destination exists until every source, hash and JSON passes.
DEST.mkdir(parents=True)
for target, data in PENDING_BYTES.items():
    output = DEST/target
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    assert output.read_bytes() == data
(DEST/'catalog.json').write_bytes(catalog_bytes)
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha(catalog_bytes)}))

"""Collect completed full-ball scalar/control mathematics once after full preflight."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-full-ball-radial-control-experiments-20261009'
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
    if source.suffix in {'.npz', '.npy', '.o', '.a', '.bin', '.rst', '.raw'} or len(data)>1048576:
        OMITTED[target] = spec
        return
    if source.suffix == '.json':
        finite(json.loads(data))
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
assert head == '0ec45fc71fcddad35393759019a1f6bc738cb4f9'
assert (HERE/'archive-README.md').is_file(), 'Missing README preflight'
assert (HERE/'independent-draft-review.json').is_file(), 'Missing independent review preflight'
frozen('continuum/full-ball-radial-assessment/immutable-full-ball-radial-advisory-20261009',
       'bff27b1f403868a8e081c0cf0b574a7bbcd75df7261bdb62053563b4df9a29fb', 'advisory')
frozen('boundary/rho-dense-mass-model-20261009/immutable-common-rho-dense-mass-scalar-model-20261009',
       '2a61499d58ceae69d0259208b790e2688156e9a6e0496f44dcd52bf27665197e', 'scalar-model')
frozen('boundary/rho-dense-mass-model-20261009/immutable-common-rho-dense-mass-root-review-20261009',
       '62d8cea2a833b86bebe0a2a1929c16b921f3f81513c84afcb6692535d335560d', 'scalar-root-review')
queue(HERE/'independent-draft-review.json', 'independent-draft-review.json')
queue(HERE/'archive-README.md', 'README.md')
queue(Path(__file__), 'collect_radial_model_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
      'verify_archive.py')
catalog = {'scope': 'Full-ball regular representation and quadrature advisory, scalar dense-mass wave energy model, and retained actual harmonic boundary principal/SAT algebra. No actual Z4c radial operator, complete CPBC, eigenvalue, evolution or stability admission.',
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

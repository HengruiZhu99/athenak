"""Collect completed core/origin and harmonic principal controls once after full preflight."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-core-principal-controls-experiments-20261009'
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
assert head == '85419e2fa74aa5cc0c357584bdb24e2604897739'
assert (HERE/'archive-README-final.md').is_file(), 'Missing README preflight'
assert (HERE/'source-math-review.json').is_file(), 'Missing independent review preflight'
frozen('boundary/total-j-flat-core-envelope-20261009/immutable-total-J-flat-core-envelope-20261009',
       'b0fde1e0eb95d6660a9fa3d190eda69207e6153c88da35366b038369ac3aa3d4', 'core')
frozen('continuum/harmonic-principal-constraint-sectors/immutable-harmonic-normal-principal-sectors-20261009',
       '05d4d7308477efe26d26ec7256fd9ba824040849363ed7f723031e5760adc0fc', 'principal')
for name in ('root-review.json', 'source-math-review.json', 'review_sources.py'):
    queue(HERE/name, 'reviews/'+name)
queue(HERE/'audit-draft.md', 'reviews/reviewed-draft.md')
queue(HERE/'archive-README.md', 'reviews/reviewed-README.md')
queue(HERE/'archive-README-final.md', 'README.md')
queue(Path(__file__), 'collect_core_principal_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
      'verify_archive.py')
catalog = {'scope': 'Exact regular Cartesian total-J flat-core polynomial envelope action, compiled local/held-out checks, and actual harmonic normal-principal4constraint+4gauge+2TT classification per sign. No radial Z4c operator, full CPBC, eigenproblem, propagation or stability result.',
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

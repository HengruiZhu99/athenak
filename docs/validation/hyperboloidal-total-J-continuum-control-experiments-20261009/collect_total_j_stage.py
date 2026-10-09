"""Collect the completed total-J angular control once after full preflight."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-total-J-continuum-control-experiments-20261009'
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
assert head == '40d843254ff96a12825f6b82ed06ee2d6853a6cf'
assert (HERE/'archive-README.md').is_file(), 'Missing README preflight'
assert (HERE/'independent-draft-review.json').is_file(), 'Missing independent review preflight'
frozen('continuum/total-j-harmonic-basis/immutable-Cartesian-total-J-basis-20261009',
       '414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e', 'basis')
frozen('boundary/total-j-local-angular-20261009/immutable-C0-spatialnorm-total-J-local-angular-20261009',
       'b4131aa02f093b7744d3de1b600e829b7213a59779672978bd84c71f6912c513', 'actual-angular')
for name in ('root-angular-review.json', 'root-build-pin-readback.json', 'independent-draft-review.json'):
    queue(HERE/name, 'root-review/'+name)
queue(HERE/'archive-README.md', 'README.md')
queue(Path(__file__), 'collect_total_j_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
      'verify_archive.py')
catalog = {'scope': 'Reviewed regular Cartesian total-J basis and actual linearized C0 physical-P/spatial-norm angular kernel control. No radial PDE operator, boundary stability, eigenvalue or finite-pulse admission.',
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

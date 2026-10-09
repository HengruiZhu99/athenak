"""Collect completed angular and Einstein-geometric Q/null jet evidence once."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-Q-angular-geometric-jet-experiments-20261009'
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
assert head == 'ef4ca08af31e133e82f88e0f4ec33b8a57681752'
gates = [('continuum/q-null-angular-ideal/immutable-Q-angular-gauge-only-ideal-20261009', 'index.json', '8c569fdf6faf0fa9afff76bebfcd8ea888c5b31734b6fb90b092188f5aab3a65', 'angular'), ('continuum/independent-q-angular-review/immutable-independent-Q-angular-gauge-review-20261009', 'index.json', '32f9738ae32e062ea8c5cf75e1160b2a6e1d50bd62b24d8888c54d776449ff37', 'angular-independent'), ('continuum/q-null-spatial-diffeo/immutable-Q-spatial-Einstein-pullback-timejet-20261009', 'index.json', 'de31bb51e55c159100a58d14acaae0f00231a7d6128bc6c3771d535b75ff5662', 'spatial'), ('continuum/independent-q-spatial-review/immutable-independent-Einstein-spatial-pullback-review-20261009', 'index.json', '8d5d3d364612b389eee2cdf1d6572097eb82a9f3c3e0e01eb2c1a2374d6c6598', 'spatial-independent')]
assert all('PENDING' not in item for row in gates for item in row)
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_q_angular_geometric_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Complete compatible angular gauge second jets at reference spatial geometry and a larger genuine Einstein spatial-pullback subset, with independent reviews. No constant sigma preserves the tested smooth quadratic-null time ideal; no finite-Omega amplitude instability, full hierarchy or evolution admission follows.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES, 'omitted_large_payloads': OMITTED,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

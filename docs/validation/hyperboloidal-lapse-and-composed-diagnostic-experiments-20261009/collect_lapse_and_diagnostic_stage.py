"""Collect completed immutable inner lapse source and composed-diagnostic attribution evidence once, preserving all source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-lapse-and-composed-diagnostic-experiments-20261009'
FILES = {}


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
    for name, spec in entries.items():
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
assert head == '1f58f7b1d18dbcf0f313f65cceb7e7e3853f7125'
gates = [
    ('continuum/inner-lapse-advection-control/immutable-inner-lapse-advection-local-20261009', 'index.json',
     '4ff923dfabbae02c509b1cfeecbe18aa173bdcd5640e6d562d1d04934c8a7e8c', 'lapse-local'),
    ('inner-lapse-advection-native/immutable-native-lapse-preflight-compact-20261009', 'index.json',
     '58a9c1a836b1b49e59f62dd1d2cc0cedbec00131c1039cb050e83aa945a8322e', 'lapse-native-preflight'),
    ('boundary/full-tensor-inner-lapse-advection/immutable-inner-lapse-global-screen-20261009', 'manifest.json',
     'fee83d35f5b666a18b6f216dd6f0e5c51beca36b470982192c4f111dd31427e3', 'lapse-global'),
    ('time-projection-controls/composed-diagnostic-attribution/immutable-composed-diagnostic-attribution-20261009', 'index.json',
     '45e12a0beba75a3b58de595d6b7843c447f4170b856a11ffb0107857379b5c0e', 'composed-diagnostic-attribution'),
]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_lapse_and_diagnostic_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed rejected finite-positive-lapse source screens and separate read-only composed constraint-functional attribution; no stable pulse, energy, exact scri or BH admission.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

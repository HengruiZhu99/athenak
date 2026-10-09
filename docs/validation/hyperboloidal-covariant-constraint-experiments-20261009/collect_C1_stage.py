"""Collect completed immutable C1 evidence once, preserving all source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-covariant-constraint-experiments-20261009'
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
assert head == 'aef47b0ab6484546887fbeef72eaa7449fff064d'
gates = [
    ('continuum/covariant-z4-candidate/immutable-C1-stiffness-20261009', 'index.json',
     '321439833808976b1b6ebfc99e443b0f61e0018b4a0aea257986ae55f0e846a8', 'stiffness-v1'),
    ('continuum/covariant-z4-candidate/immutable-C1-stiffness-v2-20261009', 'index.json',
     'd8d137e4422dec83ec685f0fee45fc80d3363cbf6a3252fec8b8d41c53c485ed', 'stiffness-v2'),
    ('continuum/c1-independent-review', 'frozen-index.json',
     '6fc5b9f35786552221f1d27c4d73b50f1bfacebd5b6e1d360b79c78b6c8a290f', 'independent-review'),
    ('covariant-native/immutable-native-C1-preflight-20261009', 'index.json',
     '5dbdc5e6c78d9de691ac44f093a480e01d2a4d7ac61cae46b27b897b998781eb', 'native-preflight'),
    ('boundary/full-tensor-covariant-c1/immutable-C1-global-screen-20261009', 'manifest.json',
     '5a44232529dd90e71f493895a6742817d73ee016066ab0cdead159dd19cdee4b', 'global-screen'),
]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_C1_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed finite-Omega C1 negative screens; no stabilization or regular scri claim.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

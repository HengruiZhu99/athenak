"""Collect completed immutable inner-C1 blend evidence once, preserving all source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-inner-constraint-blend-experiments-20261009'
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
assert head == '80b83eabffef7071ba38f914a286f99e1c4a0f35'
gates = [
    ('continuum/covariant-constraint-propagation/immutable-C1-blend-constraint-20261009', 'index.json',
     '1bb5f697691404c7a8ed19c20fc77aa79c5aa3fa31a81248d10c7e958d3ac2f0', 'tensor-subsidiary'),
    ('continuum/independent-blend-review', 'frozen-index.json',
     '4f63f6faf442045f28f995e5a74c16994526d4b10138308e6f343e51287bb2b4', 'independent-review'),
    ('blended-covariant-native/immutable-native-blend-preflight-20261009', 'index.json',
     'f8b1c9b0a2f480ab574aa11140dd98ae973bb33dc1d98915f1b2da15a6e1541b', 'native-preflight'),
    ('boundary/full-tensor-blended-c1/immutable-bulk-C1-global-screen-20261009', 'manifest.json',
     'bcbb99c769df8b4fd0dfd6ad8a8b2e3a4edfc07fcab9f48d1a5c84753ada47ee', 'global-screen'),
]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_blend_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed finite-Omega inner-C1 blend screens; no stabilization or regular scri claim.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

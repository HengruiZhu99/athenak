"""Collect completed bounded-lapse, composed-stencil and damping-vector evidence once, preserving all source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-bounded-lapse-and-stencil-experiments-20261009'
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
assert head == '38d9db1185550c3ec3659ef2392e14c46d96fb09'
gates = [('continuum/inner-conformal-trace-control/immutable-inner-conformal-trace-local-20261009', 'index.json', '801b78e9cdc7755ddda8b0ce6ea375efb02a023f7cfec420ccd8543490c150f8', 'inner-trace-local'), ('inner-trace-native/immutable-native-inner-trace-preflights-20261009', 'index.json', '6ec8049f02091e620bfa2bf95d79dc0e23490ef45169d3ea3dfd8c2d276cc706', 'inner-trace-native'), ('boundary/full-tensor-inner-trace-family/immutable-inner-trace-global-screens-v2-20261009', 'index.json', '1d72976a42b5a53b3b0df3ea1918203c14c2aab2ad2e28b9aad2244a2a54b192', 'inner-trace-global'), ('continuum/composed-global-control/immutable-composed-global-control-20261009', 'index.json', 'bad92d18a0a79b800253437f4ddf15a7d5d198ea07b3a55353647d962f11aac8', 'composed-global'), ('continuum/killing-damping-feasibility/immutable-killing-damping-feasibility-20261009', 'index.json', '7074d1a6210bbf6a9b2adcc13b5d048762edcfea9e9317ab10faf71ed9a62538', 'inertial-feasibility')]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(HERE/'collector-schema-failure/failed-collector.py','collector-schema-failure/failed-collector.py')
add(HERE/'collector-schema-failure/receipt.json','collector-schema-failure/receipt.json')
add(Path(__file__), 'collect_rejected_controls_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed rejected bounded-lapse and matching composed-Hessian controls plus mathematical inertial damping-vector feasibility; no stable pulse, energy, exact scri, puncture or BH admission.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

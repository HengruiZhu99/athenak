"""Collect completed immutable damping profile and inner calibration evidence once, preserving all source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-damping-and-inner-calibration-experiments-20261009'
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
assert head == '2392ccd1345430fe11786a7583793c2060d16f9c'
gates = [
    ('continuum/damping-profile-control/immutable-C0-profile-local-20261009', 'index.json',
     '5f12b22de8c9fb75761cf6c723d55938f4dfdc144505d3368a45b93f94d5ec4d', 'local-v1'),
    ('continuum/damping-profile-control/immutable-C0-profile-local-v2-20261009', 'index.json',
     'c9180b5bbedb2a0069a54853768a96f8b0f69736c30a8bc9c277b77f11fc39ea', 'local-v2-asbuilt-pin'),
    ('continuum/damping-profile-control/immutable-C0-profile-local-v3-20261009', 'index.json',
     'ee1bb51aa40cf7829e8f0262e4f8c6230458a1480dbecbce46071a037d757e84', 'local-v3-Nyquist-correction'),
    ('continuum/independent-damping-mapping', 'frozen-index.json',
     '3a2b38c659820243b4d850afeff0d819e4c946f01e25cf22ae6d8fb5ababd5d6', 'primary-source-algebra'),
    ('continuum/independent-profile-review', 'frozen-index.json',
     '1be97e8f5b9bda440e38acf3554495ce928917d81bba0a0026adc9fd223d9ab4', 'independent-profile-review'),
    ('continuum/independent-profile-Nyquist-review', 'frozen-index.json',
     '662a469a49890bd80decb05171dab8cb338c40bf0157d5399b048fae96c5b2dc', 'independent-Nyquist-correction'),
    ('profiled-damping-native/immutable-native-profile-preflight-20261009', 'index.json',
     'ffa0d736ff4e6a32f376771242ef25e27ff2ea094f9f975dcb5760b4520c6b09', 'native-preflight'),
    ('boundary/full-tensor-kappa2-profile/immutable-C0-profile-global-screen-20261009', 'manifest.json',
     'c767ac9a5b94a3f7df687809f402ca12623465f72ec946a3401a868191d9a4f5', 'global-screen'),
    ('continuum/inner-trumpet-calibration/immutable-inner-calibration-20261009', 'index.json',
     'bb6526a2e19ca598a4369881b642ae48258948039b060e449343b64a243c3a87', 'inner-calibration-v1'),
    ('continuum/inner-trumpet-calibration/immutable-inner-calibration-v2-20261009', 'index.json',
     '1d41a0b6f1698de614826c3a346a271c733c0d59a73a99c126b120b11cbf8774', 'inner-calibration-v2-native-span'),
]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_profile_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed finite-Omega C0 damping profile screens and analytical inner stationary calibration; no stabilization, formation or regular scri claim.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

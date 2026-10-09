"""Collect completed live damping, long-window/grid controls and scri hierarchy evidence once, preserving all source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-live-damping-and-scri-hierarchy-experiments-20261009'
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
assert head == '1959930034066ecfd43483452e723cbb197e0ad6'
gates = [('continuum/live-damping-assessment/immutable-live-math-20261009', 'index.json', '7f253e0ff6053543514831c3efb5b55cc1910aa52fdca875e6f9e9df3d40bef3', 'live-initial-bounds'), ('continuum/live-damping-control/immutable-live-damping-local-20261009', 'index.json', 'fcfd4a740fc25e999d0417598015608c009bf61de032aff5266f1f843e2d6b59', 'live-local'), ('continuum/independent-live-damping-review/immutable-independent-live-review-20261009', 'index.json', '54e403b71a9e0398fe7700cf2ab7dad836167019463e8783831563e4a28ab54d', 'live-independent-review'), ('live-damping-native/immutable-native-live-damping-preflight-20261009', 'index.json', 'a0db07fe7aad1a52afc9a609d1a0d5666f5e9e358eb51ba9ce96508cc20b252c', 'live-native-preflight'), ('boundary/full-tensor-live-damping/immutable-live-damping-global-screen-20261009', 'index.json', '4e350a8c77f1951f05dedf45ea58c3110b10010442b03031862e3f144f9317df', 'live-global-t2'), ('boundary/full-tensor-live-damping-long-window-20261009/immutable-live-damping-long-window-20261009', 'index.json', '372fb50ca797ca8bf2a43221770edd85ee450c405982c07e0aaad9fa80063c01', 'live-global-t6'), ('boundary/C0-peak-inspection-20261009', 'index.json', 'acb71c8e11fbf174b605511647b8637bd275e16b315870c55af2c1edde823817', 'C0-early-peaks'), ('boundary/full-tensor-C0-long-window-20261009/immutable-C0-long-window-20261009', 'index.json', 'd1efcbea11e730eec2e784ef7981e64f5e67d64e51bc963c6cdd7ed22834d896', 'C0-global-t6'), ('boundary/full-tensor-C0-N20-20261009/immutable-C0-N20-default-span-v2-20261009', 'index.json', '8ec4c4dd84898d19b055831696de6e3608542b1fd10a88e04f087b1f689f99aa', 'C0-N20-default'), ('boundary/full-tensor-C0-N20-phase-control-20261009/immutable-C0-N20-phase-control-20261009', 'index.json', '8a86482a1a4f0bbe3c39bdf2ddcfb8de8055984d74e9155cef45342431ba517e', 'C0-N20-phase'), ('continuum/scri-linear-hierarchy/immutable-linear-scri-hierarchy-20261009', 'index.json', 'bdaaa422b7906a7237130f146a687c1be4ed55ced61605c48f57eaa7efe4cae9', 'scri-linear-hierarchy')]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_live_long_hierarchy_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed live-damping screens, coarse projected-discrete long-window growth, grid-phase controls and actual scri first-jet hierarchy; no stable pulse, energy, exact scri closure or BH admission.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

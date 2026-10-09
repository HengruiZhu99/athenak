"""Collect completed factored Q/null source, native and rejected Cartesian evidence once."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-conformal-Q-null-feedback-experiments-20261009'
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
assert head == 'ad68e9701060cec99651368e36377f482653bc32'
gates = [('continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009', 'index.json', 'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96', 'local-core'), ('continuum/independent-q-null-feedback-review/immutable-independent-Q-null-review-20261009', 'index.json', 'a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650', 'independent-core-review'), ('q-null-native/immutable-native-Q-null-preflight-20261009', 'index.json', 'f314a5ded8966fca7f0ec6f9b8c330a2d280bcf126ffcb9800afe3ac88a059ba', 'native-preflight'), ('continuum/conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009', 'index.json', 'adaa2b1437054f4ba4cf6471eafb6f83af96a31e9ea376d723a2666abcf5e72a', 'finite-frequency-negative'), ('continuum/independent-q-followup-review/immutable-independent-Q-finite-negative-review-20261009', 'index.json', '62d9e68a59858026e09a750f981ddd820ace5d2395208ac769ce7818e8541cc4', 'finite-frequency-review'), ('boundary/full-tensor-conformal-q-null-feedback/immutable-Q-null-global-screen-20261009', 'index.json', 'd12e8e4da86f0918cfc7ad741c61a4cff3214494dc808f90df9eb061f0918214', 'global-negative')]
assert all('PENDING' not in item for row in gates for item in row)
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(Path(__file__), 'collect_q_null_stage.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed factored conformal-Q/null-feedback local and native integrity gates followed by negative finite-radius and Cartesian source/operator screens. No stable pulse, continuum instability, eigenvalue, exact-scri closure, puncture or BH admission.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES, 'omitted_large_payloads': OMITTED,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

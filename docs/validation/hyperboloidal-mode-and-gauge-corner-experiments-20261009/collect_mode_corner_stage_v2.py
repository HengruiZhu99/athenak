"""Collect completed mode, gauge-corner, C1 and repeated-reference evidence once, preserving source bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-mode-and-gauge-corner-experiments-20261009'
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
    if source.suffix in {'.npz', '.npy', '.o', '.a', '.bin', '.rst'}:
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
assert head == '269f1dbcfed80aa8959b3980ef146abaf1de6e28'
gates = [('continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009', 'index.json', '396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69', 'C0-mode'), ('boundary/full-tensor-mode-finite-step-20261009/immutable-mode-final-step-20261009', 'index.json', '3e212ec5e437c0190f687fcd3b95d52671fe814b36c9e62469924a4a01e241b0', 'mode-finite-step'), ('continuum/scri-constraint-hierarchy/immutable-constraint-Taylor-hierarchy-20261009', 'index.json', '5f013948f01b54416065ce2bd1cf182e1b61c8e8d50a84afb1264ea128925217', 'constraint-Taylor'), ('continuum/flat-penrose-height/immutable-flat-penrose-math-v2-20261009', 'index.json', 'cc8a2983de6a2e91f0ff55f4bad6928c0524fe3109e88060be4959407bf48f5e', 'flat-math-v2'), ('continuum/independent-flat-penrose-review/immutable-independent-flat-penrose-review-20261009', 'index.json', '0925f36f411bf934342f24ce76876f9d95232de5309bb665590cb2705442bbf6', 'flat-independent-review'), ('boundary/flat-penrose-reference/immutable-flat-penrose-wide-screen-20261009', 'index.json', '4e8994ea07e5f5e0095504617a135cfd7651127ad2720c57590e961bac5fe5bf', 'flat-wide-screen'), ('boundary/full-tensor-C1-long-window-20261009/immutable-fullC1-long-window-20261009', 'index.json', '836b8f1189d2be96aa994970f0c436aca594adfc449e503a954e54b73cd2335c', 'fullC1-t6'), ('continuum/discrete-mode-C1-identification/immutable-discrete-mode-C1-diagnostic-20261009', 'index.json', 'c403a4f9655e3c37ecb386c45d3e21c1f9340c2475aca5dce6e110d90d7c0ad6', 'fullC1-mode')]
for folder, index, expected, target in gates:
    frozen('build-layer-research/'+folder, index, expected, target)
add(HERE/'archive-README.md', 'README.md')
add(HERE/'audit-draft.md', 'review/audit-draft.md')
add(HERE/'boundary-review.md', 'review/boundary-review.md')
add(HERE/'first-collection-binary-exclusion/collector.py', 'collection-failure/collector.py')
add(HERE/'first-collection-binary-exclusion/receipt.json', 'collection-failure/receipt.json')
add(Path(__file__), 'collect_mode_corner_stage_v2.py')
add(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
    'verify_archive.py')
catalog = {'scope': 'Completed approximate discrete-mode and finite-step action, actual Einstein-Taylor/gauge-null corner, longer fullC1 and repeated flat-reference negative diagnostics. No certified eigenvalue, stable pulse, energy, exact-scri, puncture or BH admission.',
           'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'collection_head': head, 'files': FILES, 'omitted_large_payloads': OMITTED,
           'large_artifacts': 'Metadata/hashes only in original copied indices/receipts.'}
(DEST/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
print(json.dumps({'files': len(FILES), 'bytes': sum(x['bytes'] for x in FILES.values()),
                  'catalog_sha256': sha((DEST/'catalog.json').read_bytes())}))

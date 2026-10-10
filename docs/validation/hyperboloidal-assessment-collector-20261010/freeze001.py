"""Metadata-only wrap-up collection; never executes candidate scientific code."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / 'build-layer-research'
DEST = REPO / 'docs/validation/hyperboloidal-assessment-handoff-20261010'
ROOTS = [
    'Gaussian-a2-cache-v6-root-release-20261009',
    'Gaussian-third-jet-oracle-root-release-20261009',
    'Gaussian-third-jet-oracle-v3-root-release-20261009',
    'Gaussian-third-jet-oracle-v3-timing-root-release-20261009',
    'exact-rational-backend-source001-root-release-20261009',
    'exact-rational-backend-source002-root-release-20261009',
    'wave-map-v10-readback-root-release-20261009',
    'wave-map-v10-readback-root-release002-20261009',
    'wave-map-v9-mass-measure-diagnostic-root-release-20261009',
    'boundary/exact-rational-gauge-WIP001-independent-source-review-20261009',
    'boundary/manufactured-Gaussian-a2-cache-v6-independent-saved-cap-review-20261009',
    'boundary/manufactured-Gaussian-a2-cache-v6-independent-source-review-20261009',
    'boundary/manufactured-Gaussian-a2-cache-v7-independent-source-review-20261009',
    'boundary/manufactured-Gaussian-a2-cache-v7-independent-source-review-invocation-20261009',
    'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v10-held-20261009',
    'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v10-preparation-20261009',
    'boundary/reference-wave-map-complete-rational-rows-independent-pencil-20261009',
    'boundary/reference-wave-map-complete-rational-rows-independent-pencil-capture-20261009',
    'boundary/reference-wave-map-v10-independent-all-three-saved-review-20261009',
    'boundary/reference-wave-map-v10-independent-all-three-saved-review-invocation-20261009',
    'boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009',
    'boundary/reference-wave-map-v9-mass-measure-independent-saved-review-20261009',
    'continuum/Gaussian-a2-cache-v6-inventory-addendum-20261009',
    'continuum/exact-rational-backend-WIP001',
    'continuum/exact-rational-backend-independent-source-review-20261009',
    'continuum/exact-rational-backend-source001-held-20261009',
    'continuum/exact-rational-backend-source002-held-20261009',
    'continuum/exact-rational-backend-source002-independent-driver-review-20261009',
    'continuum/exact-rational-gauge-WIP001',
    'continuum/exact-signed-product-sum-independent-pencil-review-20261009',
    'continuum/manufactured-Gaussian-a2-Taylor-overlap-pencil-20261009',
    'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009',
    'continuum/manufactured-Gaussian-a2-cache-v7-overlap-held-20261009',
    'continuum/manufactured-Gaussian-third-jet-independent-source-review-20261009',
    'continuum/manufactured-Gaussian-third-jet-v2-independent-source-review-20261009',
    'continuum/manufactured-Gaussian-third-jet-v2-timing-failure-independent-saved-review-20261009',
    'continuum/manufactured-Gaussian-third-jet-v2-timing-mechanism-assessment-20261009',
    'continuum/manufactured-Gaussian-third-jet-v2-timing-review-runtime-addendum-20261009',
    'continuum/manufactured-Gaussian-third-jet-v2-units-independent-saved-review-20261009',
    'continuum/manufactured-Gaussian-third-jet-v3-independent-source-review-20261009',
    'continuum/manufactured-Gaussian-third-jet-v3-timing-independent-saved-review-20261009',
    'continuum/manufactured-Gaussian-third-jet-v3-units-independent-saved-review-20261009',
    'continuum/manufactured-angular-Gaussian-third-jet-oracle-held-20261009',
    'continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009',
    'continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009',
    'continuum/reference-wave-map-complete-rational-rows-pencil-20261009',
    'continuum/reference-wave-map-rational-gradient-flux-pencil-20261009',
    'continuum/exact-3x3-dual-inverse-pencil-20261009',
    'continuum/reference-wave-map-v10-final-measure-independent-source-review-20261009',
    'continuum/reference-wave-map-v9-mass-measure-independent-source-review-20261009',
]
EXPECTED = {
    'boundary/reference-wave-map-v10-independent-all-three-saved-review-20261009/index.json': '6dda5fd46f74067adf37c671bc4fe1bcc9963a93cebdf02d153f6dc5cf8d235d',
    'boundary/reference-wave-map-v10-independent-all-three-saved-review-20261009/attempt001/receipt.json': '0eeb1e98c3bdbe8644a5d50c30274875ff90091ade2b5d82ea22e3391603b939',
    'continuum/manufactured-Gaussian-third-jet-v3-timing-independent-saved-review-20261009/index.json': '1be76ee788732b801b54ef047ee9aff7367720383f20074efb2dedcae8d343fa',
    'continuum/manufactured-Gaussian-third-jet-v3-timing-independent-saved-review-20261009/receipt.json': 'd21964ec8f66df3d8493d758db55dcb6181f4e7e86d79da65bf574975bdf74ce',
    'boundary/exact-rational-gauge-WIP001-independent-source-review-20261009/index.json': '10a3a67f501584cc2a93eee919cbe02d5624faf0dfabff5ac7aafb8c389d6350',
    'boundary/exact-rational-gauge-WIP001-independent-source-review-20261009/receipt.json': '6fe5b1d896c12d972400673e1406aff4a9a7f6d6f533111dce5d4a58cf298a03',
    'continuum/exact-rational-backend-source002-held-20261009/source-index.json': 'ee41e069eee322e93d126699ec5a6251444fc843394c2c8f5d324de8f54ed5e9',
    'continuum/exact-rational-gauge-WIP001/exact_gauge_rows.hpp': 'dc5232d2ae8ff0768de98276c07c99b2ce8e82e430b4ebbe6813e9ebaba5fc1c',
    'continuum/exact-rational-backend-WIP001/exact_dyadic_ratio.hpp': '83b806893e9ed2db89a4f32dc77eba80a39d16a40774e65feb99c85102a81a2c',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

assert not DEST.exists()
assert subprocess.run(['git', 'diff', '--quiet', '27c19d20696ea6dd4704032c51dfd026218f64f2', '--', 'src', 'CMakeLists.txt'], cwd=REPO).returncode == 0
for name, expected in EXPECTED.items():
    assert sha(BASE / name) == expected, name
for root in ROOTS:
    assert (BASE / root).is_dir(), root

# Rehash the union of completed root invocations' recorded original inputs.
protected = {}
ACTUAL = [
    ('wave-map-v10-readback-root-release002-20261009', s + '-invocation001')
    for s in ['primary', 'radial_pair', 'angular_pair']
] + [
    ('Gaussian-third-jet-oracle-v3-root-release-20261009', 'units-invocation001'),
    ('Gaussian-third-jet-oracle-v3-timing-root-release-20261009', 'timing-invocation001'),
    ('wave-map-v9-mass-measure-diagnostic-root-release-20261009', 'diagnostic-invocation001'),
]
for root, attempt in ACTUAL:
    r = load(BASE / root / attempt / 'receipt.json')
    assert r['completed'] and r['returncode'] == 0 and r['inputs_unchanged'], (root, attempt)
    before = load(BASE / root / attempt / 'pins-before.json')
    after = load(BASE / root / attempt / 'pins-after.json')
    assert before == after, (root, attempt)
    for p, h in before.items():
        assert p not in protected or protected[p] == h, p
        protected[p] = h
for p, h in protected.items():
    assert sha(p) == h, p

selected = [p for root in ROOTS for p in sorted((BASE / root).rglob('*')) if p.is_file()]
sources = {}; files = {}; omitted = []; finite = 0
for p in selected:
    assert not p.is_symlink(), str(p)
    size = p.stat().st_size; digest = sha(p); sources[str(p)] = digest
    rel = p.relative_to(BASE).as_posix()
    with p.open('rb') as f:
        head = f.read(512)
    reason = None
    if p.suffix.lower() in {'.npz', '.npy', '.jsonl'}:
        reason = 'array or scientific JSONL payload; metadata only'
    elif p.name in {'probe-release.stdout', 'probe-debug.stdout', 'query-release.stdout', 'query-debug.stdout', 'native-query.stdout'}:
        reason = 'scientific query stdout duplicate; metadata only'
    elif size > 1048576:
        reason = 'file larger than 1 MiB; metadata only'
    elif p.suffix.lower() in {'.o', '.obj', '.a', '.so', '.dylib', '.dll', '.exe', '.pyc', '.pyo'} or head.startswith((b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa', b'!<arch>', b'\x93NUMPY', b'MZ')) or b'\0' in head:
        reason = 'compiled or binary payload; metadata only'
    if reason:
        omitted.append({'source': str(p), 'bytes': size, 'sha256': digest, 'reason': reason})
        continue
    data = p.read_bytes(); data.decode('utf-8')
    if p.suffix == '.json':
        load(p); finite += 1
    target = DEST / rel; target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(data)
    assert sha(target) == digest
    files[rel] = {'source': str(p), 'bytes': size, 'sha256': digest}
for p, h in sources.items():
    assert sha(p) == h, p
for p, h in protected.items():
    assert sha(p) == h, p
with (DEST / 'README.md').open('x') as f:
    f.write('''# Assessment handoff evidence, 2026-10-10\n\nThis wrap-up preserves completed private diagnostics and explicitly unexecuted prototypes at the user\'s request. See ../../hyperboloidal-assessment-handoff-20261010.md and ../../hyperboloidal-next-agent-prompt-20261010.md for status, interpretation and review priorities.\n\nAll small UTF-8 source, receipts, inputs, logs and reviews are byte-exact copies. catalog.json gives their original absolute paths, sizes and SHA256 identities. Arrays, NPY/NPZ, scientific JSONL, query-stdout duplicates, compiled/binary files and files larger than 1 MiB are metadata only. In particular the 1,276,457-byte radial v10 result and the failed partial interval payload were never decoded by this collector. Native/compiler log whitespace is preserved.\n\nThe source-only backend and whole-row reviews do not imply executable tests. Backend source002 remains held with its independent driver review unfinished. Source001 retains its registry-comparison blocker. No new Gaussian full stage, v7 interval run, exact-backend compilation, native integration or evolution was launched during wrap-up. Production src and root CMakeLists.txt remain identical to 27c19d20696ea6dd4704032c51dfd026218f64f2.\n\nThese are provenance snapshots. Original absolute paths, environment pins and omitted large local artifacts are not automatically portable. Read recipes before reconstructing a fresh, separately named attempt in another checkout. Do not edit archived snapshots, silently rebaseline dependencies, reuse old output directories, weaken tolerances or relabel earlier failures.\n''')
write(DEST / 'catalog.json', {'kind': 'user-requested assessment handoff with validated evidence and held WIP', 'roots': ROOTS, 'files': files, 'metadata_only': omitted, 'protected_inputs_rehashed': len(protected), 'all_original_inputs_unchanged': True, 'production_unchanged': True, 'scientific_runs_by_collector': 0, 'backend002_executed': False, 'whole_gauge_WIP_compiled': False})
result = {'passed': True, 'copied_files': len(files), 'metadata_omissions': len(omitted), 'finite_JSON_copies': finite, 'capsule_files': len(files) + 2, 'catalog_sha256': sha(DEST / 'catalog.json'), 'total_bytes': sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()), 'protected_inputs_rehashed': len(protected)}
write(HERE / 'freeze001.json', result)
print(json.dumps(result))

"""One-shot compact collection of completed native wave-map preflights only.

No numerical kernel, array parser, operator, evolution or result recomputation.
Existing source/log bytes are copied verbatim; omissions remain hash-indexed.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT / 'docs/validation/hyperboloidal-reference-wave-map-native-preflights-20261009'
IMPLEMENTATION = '27c19d20696ea6dd4704032c51dfd026218f64f2'
LIMIT = 1024 * 1024
ROOTS = {
    'native-preparation': ROOT / 'build-layer-research/reference-wave-map-native-held-20261009',
    'root-native-preflights': ROOT / 'build-layer-research/wave-map-native-preflight-root-20261009',
    'independent-prepared-source-review': ROOT / 'build-layer-research/boundary/reference-wave-map-native-independent-review-20261009',
    'independent-seam-snapshot-review': ROOT / 'build-layer-research/boundary/reference-wave-map-native-seam-snapshot-independent-review-20261009',
}
EXTRA_SOURCES = {
    'dependency-sources/restart_reader.py': ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py',
    'dependency-sources/restart-abi.json': ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/abi.json',
    'collection-tools/collect_preflights.py': Path(__file__).resolve(),
    'collection-tools/verify_catalog.py': HERE / 'verify_catalog.py',
}
ARRAY_SUFFIXES = {'.npy', '.npz', '.rst', '.bin', '.h5', '.hdf5', '.vtk', '.vtu'}
COMPILED_SUFFIXES = {'.o', '.a', '.so', '.dylib', '.pyc'}
MAGIC = {b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe',
         b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\x7fELF'}
FIXED = {
    ROOTS['native-preparation'] / 'analyze_snapshots.py': 'c1b9487717e85e920b274b7dcb429290eed6b0d96b3b6cbd8e00d6ea352f7c72',
    ROOTS['native-preparation'] / 'analyzer-guard-review-index.json': 'defa29818d8f8c2ce95d32018ce7bc59989083c9f1429734548d10b8b93fb882',
    ROOTS['native-preparation'] / 'probe-attempts/compile-and-seam-001/receipt.json': '54a7f3f955c294bcc45307986ec33f4d7a2fb5e10d6626c89df94a68061a2b73',
    ROOTS['native-preparation'] / 'snapshot-preflight-summary001/summary.json': '918bdcf14fa920815f219985a67746628b312833daa875b353cc23550ec95119',
    ROOTS['root-native-preflights'] / 'release.json': 'dc5787c5b579b0137cb96cea3953cdb706f22ebbd94f5e2fd12c03c1fc54ff9f',
    ROOTS['root-native-preflights'] / 'run_preflights.py': '253dc3840381f2bd413d562020f6640238dff9d0f39e33cc88ab9e9e298ac131',
    ROOTS['independent-prepared-source-review'] / 'receipt.json': '18652de97690ede671cfa8345b47258a4f72d1a3d5a38427c85efecefcb67946',
    ROOTS['independent-seam-snapshot-review'] / 'guard-revision002/receipt.json': '8b237dab197852bcb4e4b057c69cbe92a76c5b012ad827b9aa1f3a6dd0202e78',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def finite(value):
    if isinstance(value, dict):
        for x in value.values(): finite(x)
    elif isinstance(value, list):
        for x in value: finite(x)
    elif isinstance(value, float):
        assert math.isfinite(value), 'nonfinite JSON number'

def load(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    finite(value)
    return value

def dump(path, value):
    finite(value)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def relative(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(Path(path))

def checked(pins):
    for path, digest in pins.items(): assert sha(path) == digest, str(path)

def inventory():
    files = dict(EXTRA_SOURCES)
    for label, root in ROOTS.items():
        assert root.is_dir()
        for path in sorted(root.rglob('*')):
            if path.is_file():
                assert not path.is_symlink(), str(path)
                files[str(Path(label) / path.relative_to(root))] = path
    assert not any('wave-map-native-t2-root' in str(p) for p in files.values())
    return files

def production_identity():
    archive = subprocess.check_output(['git', 'archive', IMPLEMENTATION, 'src', 'CMakeLists.txt'], cwd=ROOT)
    result = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                expected = hashlib.sha256(stream.extractfile(item).read()).hexdigest()
                actual = sha(ROOT / item.name)
                assert actual == expected, ('production differs from implementation', item.name)
                result[item.name] = actual
    assert len(result) == 365
    return result

def dependency_metadata():
    paths = [ROOTS['native-preparation'] / ('build-attempts/' + mode + '-001/receipt.json')
             for mode in ['wave-map', 'c0', 'wave-map-half']]
    paths += [ROOTS['native-preparation'] / 'probe-attempts/compile-and-seam-001/receipt.json']
    union = {}
    for path in paths:
        receipt = load(path)
        for field in ['source_before', 'all_compiler_dependency_sha256']:
            for name, digest in receipt[field].items():
                assert name not in union or union[name] == digest, name
                union[name] = digest
    release = load(ROOTS['root-native-preflights'] / 'release.json')
    assert len(release['cases']) == 8
    for spec in release['cases']:
        launch_folder = ROOTS['root-native-preflights'] / 'batch001' / spec['name']
        launch = load(launch_folder / 'launch-receipt.json')
        assert launch['passed_native_process_and_provenance'] is True
        assert launch['returncode'] == 0
        assert (launch_folder / 'protected-inputs-before.json').read_bytes() == (launch_folder / 'protected-inputs-after.json').read_bytes()
        for name, digest in load(launch_folder / 'protected-inputs-before.json').items():
            assert name not in union or union[name] == digest, name
            union[name] = digest
        analysis_folder = ROOTS['native-preparation'] / 'snapshot-attempts' / (spec['name'] + '-001')
        outer = load(analysis_folder / 'receipt.json')
        inner = load(analysis_folder / 'analysis/receipt.json')
        assert outer['passed_fixed_snapshot_readback'] is True
        assert inner['passed_saved_snapshot_finite_and_diagnostic_gates'] is True
        for prefix in ['', 'analysis/']:
            assert (analysis_folder / (prefix + 'protected-inputs-before.json')).read_bytes() == (analysis_folder / (prefix + 'protected-inputs-after.json')).read_bytes()
    assert load(ROOTS['root-native-preflights'] / 'independent-audit-invocations001/receipt.json')['passed'] is True
    checked(union)
    return {name: {'sha256': digest, 'bytes': Path(name).stat().st_size,
                   'scope': 'Original dependency/source/executable metadata; not a copied runtime payload.'}
            for name, digest in sorted(union.items())}

def omit_reason(path):
    if path.suffix.lower() in ARRAY_SUFFIXES:
        return 'large_payload: every numerical array/restart/visualization payload is metadata only, regardless of size'
    if path.suffix.lower() in COMPILED_SUFFIXES:
        return 'compiled_payload: object/library/bytecode metadata only'
    with path.open('rb') as stream: header = stream.read(4)
    if header in MAGIC:
        return 'compiled_payload: executable metadata only'
    if path.stat().st_size > LIMIT:
        return 'large_payload: strict greater-than-1MiB omission'
    path.read_text(encoding='utf-8')
    return None

README = """# Completed reference-wave-map native preflights

This compact archive contains the completed eight reference/short-pulse native
controls, three private builds, actual-array seam, snapshot readbacks, prepared
source reviews and all analyzer-review history, plus root's independent saved
binary64 field audits. The source/log files are copied byte for byte. Catalog
entries retain original paths, SHA256 values and sizes, including every omitted
payload. Original receipt and review indexes are never rewritten.

The three N16/N24/N32 wave-map references and the C0 N24 reference ran to t=0.05.
The large angular pulse (.2 lapse/.1 shift) at N16/N24/N32 and small N24 pulse
(.02/.01) ran to t=0.02. All 64 saved restart arrays passed the recorded native
and independent field admission checks. These are short/reference diagnostic
gates only, with no long-time or PDE stability, exact-scri, BH-core, or
wormhole-to-trumpet acceptance claim. Production remains implementation
27c19d20696ea6dd4704032c51dfd026218f64f2. Existing prepared input/source copies
also document held long-run configurations; this collection includes no
ongoing or completed t2/long-run data from their separate prefix.

Policy: no numerical arrays, compiled executables/objects/libraries, or copied
file over 1 MiB. Every NPY/NPZ/RST/BIN is marked `large_payload` regardless of
size. All original production/compiler dependency/executable identities remain
in `dependency-metadata.json` and original receipts. This is a compact evidence
collection, not a self-contained source distribution of SDK/Kokkos dependencies.

`catalog.json` maps each archived path to its original path/hash/size and indexes
all payload omissions. `collection-receipt.json` records exact counts and the
before/after rehash. `collection-tools/verify_catalog.py` checks the compact
archive without expecting omitted payloads; `--originals` also rehashes local
originals. The original frozen-source verifiers may require their local omitted
payloads and are preserved unchanged.
"""

def main():
    started = time.monotonic()
    attempt = HERE / 'collection001'
    attempt.mkdir(parents=True, exist_ok=False)
    error = None
    try:
        checked(FIXED)
        selected = inventory()
        before = {str(p): sha(p) for p in selected.values()}
        dump(attempt / 'input-pins-before.json', before)
        deps = dependency_metadata()
        production = production_identity()
        DEST.mkdir(parents=True, exist_ok=False)
        copied = {}
        omitted = {}
        json_files = 0
        for name, source in sorted(selected.items()):
            item = {'source': relative(source), 'sha256': before[str(source)], 'bytes': source.stat().st_size}
            if source.suffix == '.json':
                load(source)
                json_files += 1
            reason = omit_reason(source)
            if reason is not None:
                item['reason'] = reason
                item['role'] = 'large_payload' if reason.startswith('large_payload') else 'compiled_payload_metadata'
                omitted[name] = item
                continue
            destination = DEST / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            assert sha(destination) == item['sha256']
            copied[name] = item
        assert selected == inventory(), 'source inventory changed during collection'
        checked(before)
        checked({p: q['sha256'] for p, q in deps.items()})
        assert production == production_identity()
        dump(attempt / 'input-pins-after.json', before)
        dump(DEST / 'dependency-metadata.json', {'scope': 'Exact original compiler/source/executable metadata; dependencies rehashed before and after compact copy.', 'files': deps})
        dump(DEST / 'production-source-identity.json', {'implementation': IMPLEMENTATION, 'count': len(production), 'all_current_bytes_equal_git_implementation': True, 'files': production})
        (DEST / 'archive-README.md').write_text(README)
        catalog = {'scope': 'Completed native preflight stage only; no ongoing long-run files, no new scientific queries or analysis.',
            'compiled_production_implementation': IMPLEMENTATION,
            'collection_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'roots': {label: relative(path) for label, path in ROOTS.items()},
            'policy': {'maximum_copied_bytes': LIMIT, 'no_arrays': True, 'no_compiled_payloads': True,
                       'all_array_suffixes_large_payload_regardless_size': sorted(ARRAY_SUFFIXES)},
            'files': copied, 'omitted_large_payloads': omitted,
            'external_dependency_metadata_sha256': sha(DEST / 'dependency-metadata.json'),
            'production_source_identity_sha256': sha(DEST / 'production-source-identity.json')}
        dump(DEST / 'catalog.json', catalog)
        for path in DEST.rglob('*'):
            if path.is_file(): assert path.stat().st_size <= LIMIT, ('archive policy size', path)
        receipt = {'passed_compact_preflight_collection': True, 'copied_files': len(copied),
            'copied_bytes': sum(x['bytes'] for x in copied.values()),
            'omitted_payload_files': len(omitted), 'omitted_payload_bytes': sum(x['bytes'] for x in omitted.values()),
            'source_inventory_files': len(selected), 'strict_finite_source_JSON_files': json_files,
            'external_dependency_identities': len(deps), 'production_files_equal_implementation': len(production),
            'catalog_sha256': sha(DEST / 'catalog.json'),
            'source_inventory_before_after_equal': True,
            'input_pins_before_sha256': sha(attempt / 'input-pins-before.json'),
            'input_pins_after_sha256': sha(attempt / 'input-pins-after.json'),
            'collector_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started,
            'new_scientific_queries': 0, 'new_native_steps': 0}
        dump(DEST / 'collection-receipt.json', receipt)
        dump(attempt / 'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as exc:
        error = repr(exc)
        dump(attempt / 'failure.json', {'passed_compact_preflight_collection': False, 'error': error,
            'collector_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started,
            'partial_destination_preserved': str(DEST)})
        raise

if __name__ == '__main__':
    main()

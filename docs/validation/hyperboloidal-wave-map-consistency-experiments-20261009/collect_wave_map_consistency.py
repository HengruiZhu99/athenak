"""One-shot byte-preserving compact archive; saved data only."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-wave-map-consistency-experiments-20261009'
MAX_BYTES = 1048576
CAPSULES = [
    ('bounded-coordinate', 'boundary/einstein-coordinate-gauge-local-20261009/immutable-bounded-inertial-coordinate-local-20261009', 'b503fccd27d951aad0503cc0bb80e0153d4172e7488b748395e4ed680c67a0ee'),
    ('principal-core', 'continuum/reference-wave-map-principal-core-20261009/immutable-principal-core-wave-map-20261009', '569ebb305b36a41fdb091330fce7da8e5be7a50f658588ca3892a51d68274ced'),
    ('nonlinear-oracle', 'continuum/immutable-nonlinear-Minkowski-wave-map-oracle-20261009', 'eb51be9c1cc698e542099169aba1bb64e8a72dd524a4cab0bd1cfb68e27b5d84'),
    ('actual-rhs', 'nonlinear-wave-map-RHS-root-20261009/immutable-nonlinear-wave-map-actual-RHS-20261009', 'c80d11d90f749e0c76a23af68a2409bb91775f0f2a5ba2cf2dc95199a6f8674f'),
    ('independent-results', 'continuum/immutable-nonlinear-wave-map-RHS-independent-review-20261009', '669e57d9457736094cbb33a705bb62d22505c91086ba85636cefa5436d485fa0'),
    ('independent-binder', 'continuum/nonlinear-wave-map-binder-independent-review-20261009', 'a447f6d236a7bdaccb1594fe13e75d723cd0220916cddbb3ff3378fbc37aa9f7'),
    ('independent-principal-core', 'boundary/reference-wave-map-principal-core-independent-20261009/immutable-independent-principal-core-review-20261009', '1c80add1b8ca8ab4abf8cb218d1e56c3c30f2bff7eac11b34442ec7e7e280492'),
]
BANNED = {'.npz', '.npy', '.jsonl', '.o', '.a', '.so', '.dylib', '.bin',
          '.rst', '.raw', '.zip', '.tar', '.gz', '.bz2', '.xz', '.pdf',
          '.png', '.jpg', '.jpeg', '.gif', '.webp'}
MAGIC = (b'\x7fELF', b'MZ', b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe',
         b'\xfe\xed\xfa\xcf', b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe',
         b'\xbe\xba\xfe\xca', b'!<arch>\n', b'PK\x03\x04', b'\x93NUMPY',
         b'\x1f\x8b', b'BZh', b'\xfd7zXZ', b'%PDF-', b'\x89PNG', b'\xff\xd8\xff')
FILES, OMITTED, PENDING, ORIGINALS, EXTERNAL = {}, {}, {}, {}, {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


def load(path):
    value = json.loads(Path(path).read_text())
    finite(value)
    return value


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p


def verify_original(path, expected, size=None, role='frozen original'):
    path = Path(path).resolve()
    assert path.is_file(), path
    assert sha(path) == expected, path
    if size is not None:
        assert path.stat().st_size == size, path
    if str(path) in ORIGINALS:
        assert ORIGINALS[str(path)]['sha256'] == expected
    ORIGINALS[str(path)] = {'sha256': expected, 'bytes': path.stat().st_size}
    if role != 'frozen original':
        EXTERNAL[str(path)] = {'sha256': expected, 'bytes': path.stat().st_size,
                               'role': role}


def queue(source, target, expected=None, size=None, force_metadata=False):
    source = Path(source).resolve()
    assert not Path(target).is_absolute() and '..' not in Path(target).parts
    assert target not in FILES and target not in OMITTED, target
    actual = sha(source)
    verify_original(source, expected or actual, size)
    data = source.read_bytes()
    reason = None
    if force_metadata:
        reason = 'original index declares external large/binary payload'
    elif len(data) > MAX_BYTES:
        reason = 'larger than 1 MiB'
    elif source.suffix.lower() in BANNED:
        reason = 'forbidden binary/array/archive suffix'
    elif data.startswith(MAGIC) or b'\x00' in data or data[257:262] == b'ustar':
        reason = 'binary or archive magic/content'
    else:
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            reason = 'non UTF-8 binary content'
    spec = {'source': str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            'sha256': actual, 'bytes': len(data)}
    if reason:
        spec['reason'] = reason
        OMITTED[target] = spec
        EXTERNAL[str(source)] = {**spec, 'role': 'metadata-only payload'}
        return
    if source.suffix == '.json':
        finite(json.loads(data))
    FILES[target] = spec
    PENDING[target] = data


def hash_dict(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, str) and len(v) == 64 and all(x in '0123456789abcdef' for x in v):
                verify_original(absolute(k), v, role='indexed source/compiler dependency')
            else:
                hash_dict(v)


def capsule(label, relative, wanted):
    base = ROOT/'build-layer-research'/relative
    index = base/'index.json'
    verify_original(index, wanted)
    idx = load(index)
    entries = idx['files']
    pairs = entries.items() if isinstance(entries, dict) else ((e['path'], e) for e in entries)
    for name, spec in pairs:
        if isinstance(spec, str):
            spec = {'sha256': spec}
        source = base/name
        metadata = spec.get('role') == 'large_payload'
        if not source.exists() and metadata:
            source = absolute(spec.get('origin', spec.get('original_path', str(base.parent/name))))
        queue(source, label+'/'+name, spec['sha256'], spec.get('bytes'), metadata)
    queue(index, label+'/index.json', wanted)
    for key in ['large_payloads_metadata_only', 'large_records']:
        for entry in idx.get(key, []):
            source = absolute(entry.get('origin', entry.get('original_path', str(base.parent/entry['path']))))
            target = label+'/'+entry['path']
            if target not in OMITTED:
                queue(source, target, entry['sha256'], entry.get('bytes'), True)
    for key in ['external_source_inputs', 'external_compiler_dependencies']:
        hash_dict(idx.get(key, {}))
    # Additional explicit metadata records in the independent readback capsule.
    metadata = base/'context-metadata.json'
    if metadata.exists():
        for entry in load(metadata):
            verify_original(absolute(entry['path']), entry['sha256'], entry['bytes'],
                            'independent context original payload')
    return {'label': label, 'path': str(index.relative_to(ROOT)), 'sha256': wanted}


def main():
    assert not DEST.exists(), 'One-shot archive already exists; never rerun into it'
    assert (HERE/'README.md').is_file() and (HERE/'verify_compact.py').is_file()
    capsules = [capsule(*entry) for entry in CAPSULES]
    addendum = ROOT/'build-layer-research/continuum/reference-wave-map-principal-core-20261009/CORE-LINEAR-SCOPE-ADDENDUM.md'
    queue(addendum, 'addenda/CORE-LINEAR-SCOPE-ADDENDUM.md',
          '0b7ce17ede6aa5be073c5ed696663814d3e9aa9b8c6e31f5faafd3b80dce5584')
    parent = ROOT/'build-layer-research/nonlinear-wave-map-RHS-root-20261009'
    frozen_empty = parent/'immutable-nonlinear-wave-map-actual-RHS-20261009/freeze.stdout'
    assert frozen_empty.read_bytes() == b''
    assert (parent/'freeze.stdout').stat().st_size > 0
    for name in ['freeze.stdout', 'freeze.stderr']:
        queue(parent/name, 'addenda/completed-owner-'+name)
    for name in ['collect_wave_map_consistency.py', 'verify_compact.py', 'README.md', 'SNAPSHOT-TIMING.md']:
        queue(HERE/name, name)
    catalog = {'scope': 'Bounded inertial-coordinate, actual harmonic principal and linear core, independent nonlinear exact-flat oracle, actual48point RHS and independent saved-data consistency controls only. No eigenproblem, propagation, native evolution, scri closure or BH acceptance.',
               'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
               'collection_HEAD': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'capsules': capsules, 'files': FILES, 'omitted_large_payloads': OMITTED,
               'external_originals_rehashed': EXTERNAL,
               'original_snapshot_timing_exception': 'The actual-RHS frozen freeze.stdout is intentionally empty at capture time. Its subsequently completed original stdout/stderr are separate additive logs; no frozen byte is rewritten.'}
    finite(catalog)
    for path, spec in ORIGINALS.items():
        assert sha(path) == spec['sha256'] and Path(path).stat().st_size == spec['bytes']
    data = (json.dumps(catalog, indent=2, allow_nan=False)+'\n').encode()
    # Destination is created only after all pins, size/magic checks and JSONs pass.
    DEST.mkdir(parents=True, exist_ok=False)
    for target, content in PENDING.items():
        p = DEST/target
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        assert p.read_bytes() == content
    (DEST/'catalog.json').write_bytes(data)
    for path, spec in ORIGINALS.items():
        assert sha(path) == spec['sha256'] and Path(path).stat().st_size == spec['bytes']
    print(json.dumps({'passed': True, 'copied_files_excluding_catalog': len(FILES),
                      'copied_bytes_excluding_catalog': sum(x['bytes'] for x in FILES.values()),
                      'catalog_sha256': hashlib.sha256(data).hexdigest(),
                      'omitted_metadata_records': len(OMITTED),
                      'external_originals_rehashed': len(EXTERNAL),
                      'all_original_inputs_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()

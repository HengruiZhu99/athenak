"""Catalog-aware compact evidence readback; omitted payloads are metadata only."""
from pathlib import Path
import hashlib
import json
import math
import sys

LIMIT = 1024 * 1024
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if isinstance(x, dict):
        for v in x.values(): finite(v)
    elif isinstance(x, list):
        for v in x: finite(v)
    elif isinstance(x, float): assert math.isfinite(x)
def load(p):
    x = json.loads(Path(p).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    finite(x)
    return x
def main():
    root = Path(sys.argv[1]).resolve()
    originals = '--originals' in sys.argv[2:]
    workspace = Path(sys.argv[sys.argv.index('--originals') + 1]).resolve() if originals else None
    c = load(root / 'catalog.json')
    r = load(root / 'collection-receipt.json')
    assert sha(root / 'catalog.json') == r['catalog_sha256']
    assert len(c['files']) == r['copied_files'] and len(c['omitted_large_payloads']) == r['omitted_payload_files']
    assert set(c['files']).isdisjoint(c['omitted_large_payloads'])
    json_count = 0
    for name, item in c['files'].items():
        p = root / name
        assert p.stat().st_size == item['bytes'] <= LIMIT
        assert sha(p) == item['sha256'], name
        assert p.suffix not in ['.npy', '.npz', '.rst', '.bin', '.o', '.a', '.so', '.dylib', '.pyc']
        if p.suffix == '.json': load(p); json_count += 1
    for name, item in c['omitted_large_payloads'].items():
        assert not (root / name).exists(), name
        assert item['role'] in ['large_payload', 'compiled_payload_metadata']
    assert sha(root / 'dependency-metadata.json') == c['external_dependency_metadata_sha256']
    assert sha(root / 'production-source-identity.json') == c['production_source_identity_sha256']
    expected = set(c['files']) | {'catalog.json', 'dependency-metadata.json', 'production-source-identity.json', 'archive-README.md', 'collection-receipt.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual == expected, actual.symmetric_difference(expected)
    for p in root.rglob('*'):
        if p.is_file(): assert p.stat().st_size <= LIMIT
    if originals:
        for item in list(c['files'].values()) + list(c['omitted_large_payloads'].values()):
            p = Path(item['source'])
            p = p if p.is_absolute() else workspace / p
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
        for name, item in load(root / 'dependency-metadata.json')['files'].items():
            p = Path(name)
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
    print(json.dumps({'passed_catalog_readback': True, 'copied_files': len(c['files']),
        'omitted_payloads': len(c['omitted_large_payloads']), 'copied_finite_JSON': json_count,
        'originals_rehashed': originals, 'catalog_sha256': sha(root / 'catalog.json')}))
if __name__ == '__main__': main()

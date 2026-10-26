#!/usr/bin/env python3
"""Extract bulk indices/coordinates from a hash-verified immutable reference.

The compact derivative artifact avoids copying multi-GB reference jets merely
to recover block ordering. Original reference payloads remain authoritative.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract(case, output):
    receipt = json.loads((case / 'receipt.json').read_text())
    checkpoint = case / 'probe_checkpoint.bin'
    payload = receipt['checkpoint_payload']
    assert checkpoint.stat().st_size == payload['bytes']
    assert sha(checkpoint) == payload['sha256']
    shapes = []
    with checkpoint.open('rb') as f:
        assert f.read(16) in [b'ATHENA_CSP_CK01!', b'ATHENA_CSP_CK02!']
        f.seek(64 + 136 + 24, 1)
        for _ in range(2):
            shape = struct.unpack('<5Q', f.read(40))
            shapes.append(shape)
            f.seek(math.prod(shape) * 8, 1)
        assert f.tell() + 8 == checkpoint.stat().st_size
    ref = case / 'cubed_sphere_reference_before.bin'
    ref_payload = receipt['reference_payloads'][ref.name]
    assert ref.stat().st_size == ref_payload['bytes']
    assert sha(ref) == ref_payload['sha256']
    counts = [s[0] * math.prod(n - 8 for n in s[2:]) for s in shapes]
    output.mkdir(parents=True, exist_ok=True)
    compact = output / 'bulk_geometry.bin'
    with ref.open('rb') as f, compact.open('wb') as g:
        magic = f.readline()
        assert magic == b'CUBED_SPHERE_REFERENCE_V1\n'
        residual = ref.stat().st_size - len(magic) - 96 - sum(math.prod(s) * 8 for s in shapes)
        assert residual % sum(counts) == 0
        cell_size = residual // sum(counts)
        assert cell_size >= 40
        count = struct.unpack('<Q', f.read(8))[0]
        shape = struct.unpack('<5Q', f.read(40))
        assert count == counts[0] and shape == shapes[0]
        for _ in range(count):
            record = f.read(40)
            assert len(record) == 40
            g.write(record)
            f.seek(cell_size - 40, 1)
    manifest = dict(format='bulk-index-xyz-le-v1', count=count, shape=shape,
                    reference_payload=ref_payload, checkpoint_payload=payload,
                    original_cell_bytes=cell_size,
                    compact_payload=dict(bytes=compact.stat().st_size, sha256=sha(compact)),
                    extractor_sha256=sha(Path(__file__)))
    (output / 'bulk_geometry.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    print(json.dumps(extract(a.case, a.output), indent=2))

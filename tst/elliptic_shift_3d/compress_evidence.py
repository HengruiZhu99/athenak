#!/usr/bin/env python3
"""Keep raw files locally and package hash-checked lossless copies for Git."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('directories', type=Path, nargs='+')
args = parser.parse_args()
for directory in args.directories:
    manifest = directory/'compressed-files.json'
    prior = json.loads(manifest.read_text()) if manifest.exists() else []
    rows = {row['original']: row for row in prior}
    ignore = directory/'.gitignore'
    patterns = ignore.read_text().splitlines() if ignore.exists() else []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix not in ('.csv', '.dat', '.checkpoint', '.bin'):
            continue
        data = path.read_bytes()
        compressed = path.with_name(path.name+'.gz')
        packed = gzip.compress(data, compresslevel=9, mtime=0)
        if gzip.decompress(packed) != data:
            raise ValueError('lossless round-trip failed: '+str(path))
        if compressed.exists() and compressed.read_bytes() != packed:
            # Older valid gzip headers may differ; never overwrite evidence.
            if gzip.decompress(compressed.read_bytes()) != data:
                raise ValueError('refusing to overwrite different evidence: '+str(compressed))
        else:
            compressed.write_bytes(packed)
        row = dict(original=path.name, compressed=compressed.name,
                   bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if path.name in rows and rows[path.name] != row:
            raise ValueError('refusing to change a frozen evidence receipt')
        rows[path.name] = row
        if '/'+path.name not in patterns:
            patterns.append('/'+path.name)
    ignore.write_text('\n'.join(patterns)+'\n')
    manifest.write_text(json.dumps(list(rows.values()), indent=2)+'\n')
    print(json.dumps(dict(directory=str(directory), checked=len(rows))))

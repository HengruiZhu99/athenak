#!/usr/bin/env python3
"""Pin the authorized completed text/scalar evidence before semantic reads."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TARGETS = [
    'docs/hyperboloidal-reference-wave-map-failure-audit.md',
    'docs/validation/hyperboloidal-reference-wave-map-partial-observations-20261009',
    'docs/validation/hyperboloidal-reference-wave-map-native-t2-failure-20261009',
    'docs/validation/hyperboloidal-reference-wave-map-native-t2-c0-N16-failure-20261009',
    'docs/validation/hyperboloidal-reference-wave-map-native-t2-wave-N24-failure-20261009',
]


def pin(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    destination = OUT / 'review-inputs-before.json'
    if destination.exists():
        raise FileExistsError(destination)
    files = []
    for name in TARGETS:
        path = ROOT / name
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file():
                if candidate.suffix.lower() in {'.rst', '.bin', '.npz', '.npy', '.jsonl'}:
                    raise RuntimeError('forbidden raw/array input: ' + str(candidate))
                files.append(pin(candidate))
    record = {'scope': 'completed document/archive source and scalar saved data only',
              'semantic_reads_started': False, 'raw_snapshot_reads': 0,
              'scientific_calls': 0, 'time_ns': time.time_ns(),
              'python': sys.version, 'files': files,
              'count': len(files), 'bytes': sum(x['bytes'] for x in files)}
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'path': str(destination), 'sha256': pin(destination)['sha256'],
                      'count': record['count'], 'bytes': record['bytes']}))


if __name__ == '__main__':
    main()

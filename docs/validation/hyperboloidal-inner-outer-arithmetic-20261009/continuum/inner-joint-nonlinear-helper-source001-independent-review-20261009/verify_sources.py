#!/usr/bin/env python3
"""Standard-library hash readback only; never import the reviewed candidate."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    before = json.loads((HERE / 'source-before.json').read_text())
    for row in before:
        payload = Path(row['path']).read_bytes()
        if (len(payload) != row['bytes'] or
                hashlib.sha256(payload).hexdigest() != row['sha256']):
            raise RuntimeError('Reviewed source/dependency drift: ' + row['path'])
    copies = json.loads((HERE / 'source-copies.json').read_text())
    for pair in copies:
        row = pair['copy']
        payload = Path(row['path']).read_bytes()
        if (len(payload) != row['bytes'] or
                hashlib.sha256(payload).hexdigest() != pair['original']['sha256']):
            raise RuntimeError('Review snapshot drift: ' + row['path'])
    print(json.dumps({'passed': True, 'protected_rows': len(before),
                      'copied_sources': len(copies), 'science_executed': False},
                     sort_keys=True))


if __name__ == '__main__':
    main()

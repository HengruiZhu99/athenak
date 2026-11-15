#!/usr/bin/env python3
"""Record exact native checkpoint metadata and bytes without changing the file."""
import argparse
import hashlib
import json
from pathlib import Path
from restart_roundtrip import parse


def receipt(path):
    path = Path(path).resolve()
    info = parse(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size,
                cycle=info['cycle'], time_native=info['time'], time_M0=3*info['time'],
                dt_native=info['dt'], blocks=info['blocks'],
                identity_sha256=hashlib.sha256(info['identity']).hexdigest(),
                identity=info['identity'].decode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(receipt(args.checkpoint), indent=2)+'\n')

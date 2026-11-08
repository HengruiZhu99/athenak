#!/usr/bin/env python3
"""Capture or verify all native build and prepared-run source inputs."""
import argparse
import hashlib
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('source', type=pathlib.Path)
parser.add_argument('manifest', type=pathlib.Path)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
source = args.source.resolve()
paths = [source/'CMakeLists.txt', source/'config.hpp.in']
if (source/'source-revision.json').is_file():
    paths.append(source/'source-revision.json')
for part in ('src', 'kokkos', 'tst/z4c_hyperboloidal'):
    paths.extend(p for p in (source/part).rglob('*')
                 if p.is_file() and '.git' not in p.parts
                 and '__pycache__' not in p.parts)
actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(paths)}
if args.check:
    expected = json.loads(args.manifest.read_text())
    changed = sorted(k for k in set(actual) | set(expected)
                     if actual.get(k) != expected.get(k))
    if changed:
        raise SystemExit('Source changed: ' + ', '.join(changed))
    print('Verified', len(actual), 'source/input files')
else:
    args.manifest.write_text(json.dumps(actual, indent=2) + '\n')
    print('Recorded', len(actual), 'source/input files')

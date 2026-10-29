#!/usr/bin/env python3
"""Explicit one-time purge of the coordinator's hash-verified archive list."""
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path('/home/hzhu/hyperbolic-core-shell-20261028')
directory = Path(__file__).resolve().parent
manifest = json.loads((directory/'manifest.json').read_text())
receipt = directory/'receipt.json'
if receipt.exists():
    raise RuntimeError('A retention receipt already exists; inspect it, do not rerun')
if (root/'.hyperbolic-debug-allocation-lock').exists():
    raise RuntimeError('Do not purge while a campaign allocation is held')
allowed = {'preflight-dissipative-v1', 'stability-dissipative-minkowski-v1'}
entries = manifest['entries']
if len(entries) != 9 or len({e['path'] for e in entries}) != 9:
    raise RuntimeError('Unexpected purge inventory')
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()
for entry in entries:
    relative = Path(entry['path'])
    if (len(relative.parts) != 4 or relative.parts[0] != 'campaigns'
            or relative.parts[1] not in allowed or relative.parts[2] != 'output'
            or not relative.name.endswith(('.checkpoint', '.failed-stage.bin'))):
        raise RuntimeError('Path is not a permitted archived payload')
    path = root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Unexpected payload link')
    campaign = root/'campaigns'/relative.parts[1]
    if not (campaign/'allocation-released.json').exists():
        raise RuntimeError('Missing terminal-release evidence')
    worker = json.loads((campaign/'worker-status.json').read_text())
    bindings = [v for case in worker['cases']
                for k,v in case.get('artifacts', {}).items() if k == entry['path']]
    if len(bindings) != 1 or any(bindings[0][key] != entry[key] for key in ('bytes','sha256')):
        raise RuntimeError('Original worker binding mismatch')
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise RuntimeError('Payload has changed since local archival')
sys.path.insert(0, str(root/'source-v11/tst/hyperbolic_shift_3d/aurora'))
import storage
record = dict(started=time.time(), status='verified_before_unlink',
              before=storage.footprint(root), manifest_sha256=digest(directory/'manifest.json'),
              entries=[dict(e, removed=False) for e in entries])
def save():
    temporary = receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(receipt)
save()
for entry in record['entries']:
    (root/entry['path']).unlink()
    entry['removed'] = True
    save()
record.update(status='completed', finished=time.time(), after=storage.footprint(root))
save()
print(json.dumps(dict(status=record['status'], files=len(entries),
                     logical_bytes=sum(e['bytes'] for e in entries), after=record['after'])))

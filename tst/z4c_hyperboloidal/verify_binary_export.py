#!/usr/bin/env python3
"""Verify the complete frozen export receipt before invoking native import."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
m=json.loads((a.directory/'manifest.json').read_text())
if m['status']!='COMPLETE':raise ValueError('incomplete export')
if hashlib.sha256(json.dumps(m['context'],sort_keys=True).encode()).hexdigest()!=m['identity']:
    raise ValueError('export context identity mismatch')
if set(m['blocks'])!={p.name for p in a.directory.glob('*.hbi')}:raise ValueError('block inventory mismatch')
for name,record in m['blocks'].items():
    h=hashlib.sha256()
    with (a.directory/name).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=record['sha256']:raise ValueError('block changed: '+name)
    with (a.directory/name).open() as f:header=f.readline().split()
    if header[0]!='HYPERBOL_BINARY_V1' or header[1]!=m['identity'] or tuple(map(int,header[3:]))!=(record['bulk'],record['scri']):
        raise ValueError('block header identity/count mismatch: '+name)
print('PASS',len(m['blocks']),'blocks; identity',m['identity'])

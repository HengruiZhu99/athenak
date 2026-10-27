#!/usr/bin/env python3
"""Restore losslessly compressed pilot CSV/initializer files, checking hashes."""
import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
root=a.root.resolve();checked=0
for manifest in root.rglob('compressed-files.json'):
 for item in json.loads(manifest.read_text()):
  original=(manifest.parent/item['original']).resolve();compressed=(manifest.parent/item['compressed']).resolve()
  if root not in original.parents or root not in compressed.parents:raise ValueError('Archive path leaves evidence root')
  data=gzip.decompress(compressed.read_bytes())
  if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError(f'Archive checksum mismatch: {compressed}')
  if original.exists():
   if hashlib.sha256(original.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'Refusing to overwrite different file: {original}')
  elif not a.check_only:original.write_bytes(data)
  checked+=1
print(json.dumps(dict(checked=checked,restored=not a.check_only)))

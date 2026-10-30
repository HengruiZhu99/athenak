#!/usr/bin/env python3
"""Verify a directly streamed evidence archive and unpack regular files only."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1<<20),b''):h.update(data)
 return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('receipt',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
r=json.loads(a.receipt.read_text());assert r['remote_archive_created'] is False
assert a.archive.stat().st_size==r['bytes'] and digest(a.archive)==r['sha256']
a.destination.mkdir(parents=True,exist_ok=True)
with tarfile.open(a.archive,'r:gz') as t:
 members=t.getmembers();names=set()
 for m in members:
  path=PurePosixPath(m.name)
  assert not path.is_absolute() and '..' not in path.parts and m.name not in names
  assert m.isfile() or (m.islnk() and m.linkname in names)
  assert not (a.destination/m.name).is_symlink()
  names.add(m.name)
 for m in members:
  target=a.destination/m.name
  assert all(not q.is_symlink() for q in [target,*target.parents])
  target.parent.mkdir(parents=True,exist_ok=True)
  with t.extractfile(m) as source,target.open('wb') as out:shutil.copyfileobj(source,out,1<<20)
print(json.dumps(dict(archive_verified=True,archive_bytes=r['bytes'],archive_sha256=r['sha256'],files=len(members))))

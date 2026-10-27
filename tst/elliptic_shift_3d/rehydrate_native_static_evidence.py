#!/usr/bin/env python3
"""Restore the exact native static-SMR evidence bytes, with hash/path guards."""
import argparse,hashlib,json,shutil,tarfile,tempfile
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('package',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--sources',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args();package=a.package.resolve();repository=package.parents[3];destination=a.destination.resolve() if a.destination else repository;index=json.loads((package/'chunk-index.json').read_text());archive=package/index['archive'];assert digest(archive)==index['archive_sha256'];temporary=Path(tempfile.mkdtemp(prefix='native-static-rehydrate-'));checked=0;source_checked=0
 def target(relative):
  path=(destination/relative).resolve()
  if not path.is_relative_to(destination):raise ValueError('archive path leaves destination')
  return path
 try:
  expected={r['sha256']:r['bytes'] for r in index['unique_chunks']};seen=set()
  with tarfile.open(archive,'r:gz') as tar:
   for member in tar:
    if not member.isfile() or member.name not in expected:raise ValueError('unexpected/nonregular data chunk')
    data=tar.extractfile(member).read()
    if len(data)!=expected[member.name] or hashlib.sha256(data).hexdigest()!=member.name:raise ValueError('corrupt data chunk')
    (temporary/member.name).write_bytes(data);seen.add(member.name)
  assert seen==set(expected)
  for row in index['files']:
   path=target(row['path']);h=hashlib.sha256();size=0
   for key in row['chunks']:
    data=(temporary/key).read_bytes();h.update(data);size+=len(data)
   assert size==row['bytes'] and h.hexdigest()==row['sha256']
   if path.exists():assert digest(path)==row['sha256'],'refusing different existing evidence'
   elif not a.check_only:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as out:
     for key in row['chunks']:out.write((temporary/key).read_bytes())
   checked+=1
  if a.sources:
   manifest=json.loads((package/'manifest.json').read_text())
   for directory in manifest['directories']:
    folder=repository/directory['directory'];records=json.loads((folder/'source-archives.json').read_text())
    for row in records:
     path=folder/row['archive'];assert digest(path)==row['sha256'];members={r['path']:r for r in row['members']};seen=set()
     with tarfile.open(path,'r:gz') as tar:
      for member in tar:
       if not member.isfile() or member.name not in members:raise ValueError('unexpected/nonregular source member')
       expected=members[member.name];data=tar.extractfile(member).read();assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'];out=target(Path(directory['directory'])/row['original_directory']/member.name)
       if out.exists():assert digest(out)==expected['sha256'],'refusing different existing source'
       elif not a.check_only:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
       seen.add(member.name);source_checked+=1
     assert seen==set(members)
 finally:shutil.rmtree(temporary)
 print(json.dumps({'scientific_files_checked':checked,'source_members_checked':source_checked,'restored':not a.check_only,'all_hashes_match':True}))
if __name__=='__main__':main()

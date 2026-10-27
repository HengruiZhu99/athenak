#!/usr/bin/env python3
"""Exact-byte deduplication of regular native-adaptive scientific outputs."""
import argparse,gzip,hashlib,io,json,shutil,tarfile,tempfile,time
from pathlib import Path
sha=lambda data:hashlib.sha256(data).hexdigest()
p=argparse.ArgumentParser();p.add_argument('package',type=Path);a=p.parse_args();root=Path.cwd();out=a.package.resolve();original=(out/'manifest.json').read_bytes();(out/'manifest-perfile-gzip.json').write_bytes(original);m=json.loads(original);seen=set();files=[];chunks=[];chunk_size=262144;target=out/'data-chunks.tar.gz';assert not target.exists();start=time.monotonic()
with target.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,compresslevel=6,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as archive:
 for directory in m['directories']:
  folder=root/directory['directory'];records=json.loads((folder/'compressed-files.json').read_text())
  for row in records:
   path=folder/row['original'];data=path.read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256'];identifiers=[];first=len(data)%chunk_size;offset=0
   # End-aligned chunks preserve sharing of identical native field payloads
   # even when restart/output headers have different pathname lengths.
   while offset<len(data):
    count=first if offset==0 and first else chunk_size;part=data[offset:offset+count];offset+=len(part);key=sha(part);identifiers.append(key)
    if key not in seen:
     info=tarfile.TarInfo(key);info.size=len(part);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(part));seen.add(key);chunks.append({'sha256':key,'bytes':len(part)})
   files.append({'path':str(path.relative_to(root)),'bytes':len(data),'sha256':row['sha256'],'chunks':identifiers})
  patterns=set((folder/'.gitignore').read_text().splitlines());patterns.update('/'+r['compressed'] for r in records);(folder/'local-gzip-files.json').write_bytes((folder/'compressed-files.json').read_bytes());patterns.add('/compressed-files.json');(folder/'.gitignore').write_text('\n'.join(sorted(patterns))+'\n');print(folder.name,'chunked',len(records),flush=True)
index={'scope':'Byte-exact content-addressed scientific output capsule; all originals and per-file gzip remain local; source archives are separate','chunk_bytes':chunk_size,'alignment':'first partial chunk then fixed-size chunks to file end','archive':target.name,'archive_sha256':sha(target.read_bytes()),'unique_chunks':chunks,'files':files,'contains_executables_or_objects':False};(out/'chunk-index.json').write_text(json.dumps(index,indent=2)+'\n')
cache=Path(tempfile.mkdtemp(prefix='native-evidence-verify-',dir=out))
try:
 with tarfile.open(target,'r:gz') as archive:
  for member in archive:
   data=archive.extractfile(member).read();assert member.name==sha(data);(cache/member.name).write_bytes(data)
 for row in files:
  digest=hashlib.sha256();count=0
  for key in row['chunks']:
   data=(cache/key).read_bytes();digest.update(data);count+=len(data)
  assert count==row['bytes'] and digest.hexdigest()==row['sha256']
finally:shutil.rmtree(cache)
# Replace Git-facing inventory: gzip copies are represented exactly by capsule.
for directory in m['directories']:
 folder=root/directory['directory'];records=json.loads((folder/'compressed-files.json').read_text());omitted={r['compressed'] for r in records};kept=[]
 for row in directory['package_files']:
  if row['path'] in omitted or row['path']=='compressed-files.json':continue
  data=(folder/row['path']).read_bytes();kept.append({'path':row['path'],'bytes':len(data),'sha256':sha(data)})
 local=folder/'local-gzip-files.json';kept.append({'path':local.name,'bytes':local.stat().st_size,'sha256':sha(local.read_bytes())});directory['package_files']=kept;directory['data_packaging']='chunk-index.json + data-chunks.tar.gz in package directory; per-file gzip is local only'
m['local_gzip_metadata_note']='local-gzip-files.json describes local-only gzip copies; use the native capsule rehydrator rather than generic gzip restoration.';m['deduplicated_scientific_data']={'index':'chunk-index.json','index_sha256':sha((out/'chunk-index.json').read_bytes()),'archive':target.name,'archive_sha256':index['archive_sha256'],'files':len(files),'unique_chunks':len(chunks),'raw_bytes':sum(r['bytes'] for r in files),'compressed_archive_bytes':target.stat().st_size,'all_files_roundtrip_verified':True};m['total_package_files']=sum(len(r['package_files']) for r in m['directories'])+2;m['total_package_bytes']=sum(f['bytes'] for r in m['directories'] for f in r['package_files'])+target.stat().st_size+(out/'chunk-index.json').stat().st_size;m['deduplication_seconds']=time.monotonic()-start;(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'package_bytes':m['total_package_bytes'],'archive_bytes':target.stat().st_size,'scientific_files':len(files),'chunks':len(chunks),'seconds':m['deduplication_seconds'],'manifest_sha256':sha((out/'manifest.json').read_bytes())}))

#!/usr/bin/env python3
"""Split the byte-exact capsule into <=32MiB Git-facing parts; originals stay local."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('package',type=Path);a=p.parse_args();out=a.package.resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();index=json.loads((out/'chunk-index.json').read_text());archive=out/index['archive'];parts=[]
with archive.open('rb') as stream:
 while True:
  data=stream.read(32*1024*1024)
  if not data:break
  target=out/(archive.name+f'.part{len(parts):03d}');assert not target.exists();target.write_bytes(data);parts.append({'path':target.name,'bytes':len(data),'sha256':sha(target)})
index['archive_parts']=parts;index['archive_parts_note']='Concatenate in listed order to reproduce the exact gzip archive; individual fragments are not standalone gzip files.';(out/'chunk-index.json').write_text(json.dumps(index,indent=2)+'\n');m=json.loads((out/'manifest.json').read_text());m['deduplicated_scientific_data'].update(index_sha256=sha(out/'chunk-index.json'),archive_parts=parts,unsplit_archive_local_only=True);m['total_package_files']+=len(parts)-1;m['total_package_bytes']=sum(f['bytes'] for d in m['directories'] for f in d['package_files'])+sum(r['bytes'] for r in parts)+(out/'chunk-index.json').stat().st_size;(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(out/'.gitignore').write_text('/'+archive.name+'\n');print(json.dumps({'parts':len(parts),'maximum_part_bytes':max(r['bytes'] for r in parts),'total_package_bytes':m['total_package_bytes'],'manifest_sha256':sha(out/'manifest.json')}))

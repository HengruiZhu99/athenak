#!/usr/bin/env python3
"""Portable content-addressed source tar; excludes Git pointers and AppleDouble."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from aurora_resource_policy import prepare
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extra',nargs='*',default=[]);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing archive is immutable')
prepare(a.prepared)
files=sorted(f for f in (a.prepared/'source').rglob('*') if f.is_file() and f.name!='.git' and not f.name.startswith('._'))
files += [a.prepared/k for k in ['run_aurora.pbs','cpu-admission.json','resource-policy.json',*a.extra]]
manifest={str(f.relative_to(a.prepared)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(a.prepared/'portable-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(a.prepared/'portable-source.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in manifest.items()))
with tarfile.open(a.output,'w:gz',format=tarfile.USTAR_FORMAT) as archive:
 for f in files:
  info=tarfile.TarInfo(str(f.relative_to(a.prepared)));info.size=f.stat().st_size;info.mode=0o644;info.mtime=int(f.stat().st_mtime)
  with f.open('rb') as data:archive.addfile(info,data)
 for file,name in [('portable-source-manifest.json','source-manifest.json'),('portable-source.sha256','source.sha256')]:
  f=a.prepared/file;info=tarfile.TarInfo(name);info.size=f.stat().st_size;info.mode=0o644
  with f.open('rb') as data:archive.addfile(info,data)
print(json.dumps(dict(files=len(files),archive_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),archive_bytes=a.output.stat().st_size)))

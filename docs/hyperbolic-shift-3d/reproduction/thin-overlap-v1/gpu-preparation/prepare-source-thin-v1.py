#!/usr/bin/python3.11
from pathlib import Path
import hashlib,json,os,sys,tarfile
root=Path('/home/hzhu/hyperbolic-core-shell-20261028')
sys.path.insert(0,str(root/'source-v14/tst/hyperbolic_shift_3d/aurora'))
import storage
assert not (root/'source-thin-v1').exists()
assert not (root/'build-sycl-mpi-thin-v1').exists()
assert not (root/'source-thin-v1-manifest.json').exists()
assert storage.footprint(root)['bytes']+450000000<5000000000
source=json.loads((root/'source-thin-v1-non-kokkos-manifest.json').read_text())
provenance=json.loads((root/'source-thin-v1-provenance.json').read_text())
archive=root/'hyperbolic-aurora-source-thin-v1.tar.gz'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(archive)==provenance['archive_sha256']
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers()
 assert len(members)==len(source) and {m.name for m in members}==set(source)
 assert all(m.isfile() and m.name.startswith('source-thin-v1/') and '..' not in Path(m.name).parts for m in members)
 tar.extractall(root)
for f,wanted in source.items():assert digest(root/f)==wanted,f
old=json.loads((root/'source-v14-manifest.json').read_text());count=0
for f,wanted in old.items():
 if not f.startswith('source-v14/kokkos/'):continue
 donor=root/f;assert donor.is_file() and not donor.is_symlink() and digest(donor)==wanted,f
 new=f.replace('source-v14/','source-thin-v1/',1);destination=root/new
 destination.parent.mkdir(parents=True,exist_ok=True)
 os.link(donor,destination);source[new]=wanted;count+=1
assert count==1486
manifest=root/'source-thin-v1-manifest.json'
manifest.write_text(json.dumps(source,sort_keys=True,indent=2)+'\n')
for f,wanted in source.items():assert digest(root/f)==wanted,f
print(json.dumps(dict(status='passed',source_commit=provenance['source_commit'],source_files=len(source),kokkos_files=count,source_manifest_sha256=digest(manifest),footprint=storage.footprint(root))))

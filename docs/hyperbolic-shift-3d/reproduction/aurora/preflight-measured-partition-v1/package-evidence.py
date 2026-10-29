from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/hzhu/hyperbolic-core-shell-20261028')
c=root/'campaigns/preflight-measured-partition-v1'
assert (c/'allocation-released.json').is_file()
archive=root/'preflight-measured-partition-v1-evidence.tar.gz';assert not archive.exists()
import sys
sys.path.insert(0,str(root/'source-v22/tst/hyperbolic_shift_3d/aurora'))
import storage
# Upper bound exceeds the prior 132 MB compressed evidence archive.
assert storage.footprint(root)['bytes']+300000000<5000000000
seen={}
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
 for p in sorted(c.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink()
  name=str(p.relative_to(c));digest=hashlib.sha256(p.read_bytes()).hexdigest()
  # Encode verified byte-identical checkpoints once, with a tar hardlink.
  if p.suffix=='.checkpoint' and digest in seen:
   info=t.gettarinfo(str(p),arcname=name);info.type=tarfile.LNKTYPE;info.linkname=seen[digest];info.size=0;t.addfile(info)
  else:
   t.add(p,arcname=name,recursive=False)
   if p.suffix=='.checkpoint':seen[digest]=name
print(json.dumps(dict(path=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))

from pathlib import Path
import hashlib,json,tarfile,sys
root=Path('/home/hzhu/hyperbolic-core-shell-20261028')
c=root/'campaigns/preflight-radial-field-batch-v1'
assert (c/'allocation-released.json').is_file()
sys.path.insert(0,str(root/'source-v24/tst/hyperbolic_shift_3d/aurora'));import storage
assert storage.footprint(root)['bytes']<5000000000
class Writer:
 def __init__(self):self.hash=hashlib.sha256();self.size=0
 def write(self,data):
  self.hash.update(data);self.size+=len(data);return sys.stdout.buffer.write(data)
 def flush(self):sys.stdout.buffer.flush()
w=Writer();seen={}
with tarfile.open(fileobj=w,mode='w:gz',compresslevel=3) as t:
 for p in sorted(c.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink()
  name=str(p.relative_to(c));digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if p.suffix=='.checkpoint' and digest in seen:
   info=t.gettarinfo(str(p),arcname=name);info.type=tarfile.LNKTYPE;info.linkname=seen[digest];info.size=0;t.addfile(info)
  else:
   t.add(p,arcname=name,recursive=False)
   if p.suffix=='.checkpoint':seen[digest]=name
w.flush()
print(json.dumps(dict(path='preflight-radial-field-batch-v1-evidence.tar.gz',bytes=w.size,sha256=w.hash.hexdigest(),remote_archive_created=False)),file=sys.stderr)

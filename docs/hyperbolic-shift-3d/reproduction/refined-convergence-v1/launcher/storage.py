import os
from pathlib import Path
LIMIT=50_000_000_000
PRIOR=[Path('/home/hzhu/hyperboloidal-performance-scaling-20261103'),Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hyperboloidal-refined-20261103-v1'),Path('/home/hzhu/hyperbolic-core-shell-20261028')]
def capacity(root,planned=0,reserve=2_000_000_000):
 total=0;seen=set()
 for base in PRIOR+[Path(root)]:
  if not base.is_dir():raise RuntimeError('Missing accounted root '+str(base))
  for directory,children,names in os.walk(base,followlinks=False):
   for p in [Path(directory)]+[Path(directory)/x for x in names]+[Path(directory)/x for x in children if (Path(directory)/x).is_symlink()]:
    q=p.lstat();k=(q.st_dev,q.st_ino)
    if k in seen:continue
    seen.add(k);total+=max(q.st_size,q.st_blocks*512)
 r=dict(bytes=total,limit_bytes=LIMIT,planned_bytes=planned,reserve_bytes=reserve,headroom_bytes=LIMIT-total-planned-reserve,roots=list(map(str,PRIOR+[Path(root)])))
 if r['headroom_bytes']<0:raise RuntimeError('Storage cap exceeded '+str(r))
 return r

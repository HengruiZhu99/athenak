"""Fail-closed allocated/logical byte accounting, without duplicate tree walks.

Count each (device,inode) once, including directories and symlinks themselves;
never follow symlink children. Root paths are all required to exist as directories.
This preserves the frozen storage cap/roots/reservations; only traversal changes.
"""
import os,stat
from pathlib import Path
LIMIT=50_000_000_000
PRIOR=[Path('/lus/flare/projects/CompactBinaryMerger/hzhu/refined-convergence-20261104-v1'),Path('/home/hzhu/hyperboloidal-performance-scaling-20261103'),Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hyperboloidal-refined-20261103-v1'),Path('/home/hzhu/hyperbolic-core-shell-20261028')]
def accounted_bytes(roots):
 roots=[Path(p) for p in roots]
 for p in roots:
  if not p.is_dir():raise RuntimeError('Missing accounted root '+str(p))
 seen=set();total=0;directories=0;entries=0
 # A symlink root is counted as a link AND traversed, matching os.walk(root).
 # Children which are symlinks remain objects only, never followed.
 traversed=set()
 for root in roots:
  q=root.lstat();key=(q.st_dev,q.st_ino)
  if key not in seen:seen.add(key);total+=max(q.st_size,q.st_blocks*512)
  pending=[root]
  while pending:
   p=pending.pop();q=p.stat();key=(q.st_dev,q.st_ino)
   if key in traversed:continue
   traversed.add(key);directories+=1
   with os.scandir(p) as iterator:
    for entry in iterator:
     entries+=1;q=entry.stat(follow_symlinks=False);key=(q.st_dev,q.st_ino)
     if key not in seen:seen.add(key);total+=max(q.st_size,q.st_blocks*512)
     if stat.S_ISDIR(q.st_mode):pending.append(Path(entry.path))
 return dict(bytes=total,unique_inodes=len(seen),directories_scanned=directories,entries_scanned=entries)
def capacity(root,planned=0,reserve=2_000_000_000):
 roots=PRIOR+[Path(root)];scan=accounted_bytes(roots);total=scan['bytes']
 r=dict(bytes=total,limit_bytes=LIMIT,planned_bytes=planned,reserve_bytes=reserve,headroom_bytes=LIMIT-total-planned-reserve,roots=list(map(str,roots)),scan=scan)
 if r['headroom_bytes']<0:raise RuntimeError('Storage cap exceeded '+str(r))
 return r

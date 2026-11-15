"""Read-only complete accepted-state comparison with the prior 26 M0 run."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
case=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/n12-full-scri-20261115')
sys.path.insert(0,str(case/'analysis-source/tst/z4c_hyperboloidal'))
from restart_roundtrip import parse
old=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111/science/N12-full-overlap-30M-20261114/segment-9089673/rst/headon.00026.rst')
a=parse(old);matches=[]
for p in case.glob('segment-*/rst/*.rst'):
 try:b=parse(p)
 except (AssertionError,ValueError):continue
 if b['cycle']==a['cycle']:matches.append(b)
if not matches:
 print('WAITING_FOR_MATCHING_CHECKPOINT',a['cycle']);sys.exit(2)
assert len(matches)==1
b=matches[0]
for key in ['blocks','record_blocks','cycle','time','dt','atlas','topology','identity','stride']:
 assert a[key]==b[key],key
ha=hashlib.sha256();hb=hashlib.sha256();different=0;maximum=0.;first=None;size=0
with old.open('rb') as f,Path(b['path']).open('rb') as g:
 f.seek(a['payload_at']);g.seek(b['payload_at'])
 while True:
  x=f.read(8*1024*1024);y=g.read(8*1024*1024)
  assert len(x)==len(y)
  if not x:break
  ha.update(x);hb.update(y)
  if x!=y:
   p=np.frombuffer(x,dtype='=f8');q=np.frombuffer(y,dtype='=f8');bad=p.view('=u8')!=q.view('=u8')
   if first is None:first=int(size//8+np.flatnonzero(bad)[0])
   different+=int(np.sum(bad));maximum=max(maximum,float(np.max(abs(p-q))))
  size+=len(x)
report=dict(reference=str(old),candidate=b['path'],cycle=a['cycle'],time_M0=3*a['time'],payload_bytes=size,reference_payload_sha256=ha.hexdigest(),candidate_payload_sha256=hb.hexdigest(),bitwise_equal=different==0,differing_scalars=different,first_differing_scalar=first,max_abs_difference=maximum)
(case/'long-trajectory-26M.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

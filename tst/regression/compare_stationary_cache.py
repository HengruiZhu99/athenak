"""Compare complete cached/uncached restart payloads, allowing only the cache flag.
Usage: python compare_stationary_cache.py LEFT RIGHT OUTPUT_JSON RANKS
"""
import sys,json,hashlib,os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'analysis/tde_revalidation'))
from validate_checkpoint import read_header,equal_partition
left,right,out=map(Path,sys.argv[1:4]);records=[];headers=[];ranks=int(sys.argv[4])
for directory in (left,right):
 h=max([read_header(p) for p in (directory/'rst/rank_00000000').glob('*.rst')],key=lambda h:h['cycle']);headers.append(h)
assert headers[0]['time']==headers[1]['time'] and headers[0]['cycle']==headers[1]['cycle'],'Different actual stop times; cannot compare final payloads'
for key in ['total','stride','indices','root_indices','region','locations','costs']:
 assert np.array_equal(headers[0][key],headers[1][key]),'Geometry/format differs: '+key
params=[]
for h in headers:
 p={s:dict(v) for s,v in h['params'].items()};p['problem'].pop('cache_stationary_background',None);params.append(p)
assert params[0]==params[1],'Inputs differ beyond cache option'
assert np.all(np.array(headers[0]['costs'])==1),'Requires unit-cost ownership'
_,counts=equal_partition(headers[0]['total'],ranks)
for directory,h in zip((left,right),headers):
 name=Path(h['path']).name
 assert set((directory/'rst').glob('rank_*/'+name))==set(directory/'rst'/('rank_%08d'%r)/name for r in range(ranks)),'Incomplete/extra rank cohort'
for rank in range(ranks):
 digests=[]
 for directory,h in zip((left,right),headers):
  p=directory/'rst'/('rank_%08d'%rank)/Path(h['path']).name;before=p.stat();hr=read_header(p)
  assert hr['header']==h['header'],'Inconsistent rank header'
  assert before.st_size==hr['payload_start']+counts[rank]*hr['stride'],'Truncated/extra payload'
  payload=np.memmap(str(p),mode='r',dtype='<f8',offset=hr['payload_start']);assert np.isfinite(payload).all();del payload
  digest=hashlib.sha256()
  with p.open('rb') as f:
   f.seek(hr['payload_start'])
   for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
  after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Output changed during comparison';digests.append(digest.hexdigest())
 assert digests[0]==digests[1],'Payload mismatch at rank%d'%rank
 records.append({'rank':rank,'payload_sha256':digests[0]})
result={'passed':True,'scope':'Same-binary cache off/on complete finite checkpoint payload equality, headers and geometry; not independent metric or stability audit','time':headers[0]['time'],'cycle':headers[0]['cycle'],'blocks':headers[0]['total'],'ranks':ranks,'records':records};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))

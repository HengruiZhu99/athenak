"""Compare native RF2 SMR state after MPI2 -> MPI1 restart, without evolution."""
from pathlib import Path
import hashlib,json,struct
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists())
OUT=ROOT/'output/refined-convergence'
def read(name):
 p=OUT/(name+'.checkpoint'); b=p.read_bytes(); at=b.index(b'\0')+1
 tag=b[:at-1].decode(); assert tag.startswith('HYP3D_AMR_RF2;')
 def get(fmt):
  nonlocal at
  q=struct.unpack_from('<'+fmt,b,at);at+=struct.calcsize('<'+fmt);return q[0] if len(q)==1 else q
 size=get('Q'); identity=b[at:at+size].decode();at+=size
 for _ in range(31): at=b.index(b'\0',at)+1
 config=get('iiiQQ');time,dt,seed=get('ddd');tracker=get('3d');floors=get('8d');diagnostics=get('8d')
 count=get('Q'); topology=b[at:at+count*16];at+=count*16;ages=b[at:at+count*4];at+=count*4
 fields=[]
 for _ in range(8):
  n=get('Q');fields.append(np.frombuffer(b,dtype='<f8',count=n,offset=at));at+=n*8
 assert get('Q')==0x414d5231434f4d50 and at+8==len(b)
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),identity=identity,config=config,time=time,dt=dt,seed=seed,tracker=tracker,floors=floors,diagnostics=diagnostics,leaves=count),topology,ages,fields
u,ut,ua,uf=read('smr-policy-uninterrupted2');r,rt,ra,rf=read('smr-policy-resume2')
assert u['identity']==r['identity'] and u['config']==r['config'] and ut==rt and ua==ra
assert u['time']==r['time'] and u['dt']==r['dt']
comparisons=[]
for i,(a,b) in enumerate(zip(uf,rf)):
 assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 error=float(np.max(np.abs(a-b)/(1+np.maximum(np.abs(a),np.abs(b))))) if a.size else 0
 comparisons.append(dict(array=i,elements=a.size,scaled_max=error,bit_identical=bool(np.array_equal(a,b))))
assert max(c['scaled_max'] for c in comparisons)<1e-12
rejected=(OUT/'smr-policy-rejected.log').read_text(); assert 'checkpoint formulation identity mismatch' in rejected
assert not (OUT/'smr-policy-rejected.checkpoint').exists()
result=dict(passed=True,uninterrupted=u,restarted=r,arrays=comparisons,policy_mismatch_rejected=True,qualification='Two stationary steps at N12; MPI2 uninterrupted vs MPI2 step1 then MPI1 step2. Not long evolution, convergence, GPU or moving AMR validation.')
(Path(__file__).resolve().parent/'smr-restart-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

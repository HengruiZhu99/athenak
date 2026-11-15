"""Predeclared checks for changed floating-point reduction order; no physics changes."""
from pathlib import Path
import struct,hashlib
import numpy as np
# Fixed BEFORE inspecting candidate results. Normalize each column separately.
TOLERANCES={'cpu2':5e-12,'short':1e-10,'long':1e-9,'history_relative':1e-8}
EXACT_COLUMNS=[33,34,35,38] # floor/nonpositive/nonfinite counts and first-floor step

def offset(p):
 with Path(p).open('rb') as f:
  text=bytearray()
  while not text.endswith(b'<par_end>\n'):
   c=f.read(1);assert c;text+=c
  nmb=struct.unpack_from('=i',f.read(struct.calcsize('=ii9d38i2di')))[0]
  f.seek(struct.unpack('=Q',f.read(8))[0]+nmb*24,1)
  meta=struct.unpack('=8Q',f.read(64));identity=f.read(meta[2]);stride=struct.unpack('=Q',f.read(8))[0]
  return f.tell(),identity,stride

def compare(p,q,kind):
 p,q=Path(p),Path(q);po,pi,ps=offset(p);qo,qi,qs=offset(q)
 assert pi==qi and ps==qs and p.stat().st_size-po==q.stat().st_size-qo
 diff=np.zeros(39);scale=np.ones(39);bits=True;count=0;hd=[hashlib.sha256(),hashlib.sha256()]
 with p.open('rb') as x,q.open('rb') as y:
  x.seek(po);y.seek(qo)
  while True:
   a,b=x.read(39*8*16384),y.read(39*8*16384)
   assert len(a)==len(b)
   if not a:break
   hd[0].update(a);hd[1].update(b);bits &= a==b
   u=np.frombuffer(a,dtype='=f8').reshape(-1,39);v=np.frombuffer(b,dtype='=f8').reshape(-1,39)
   assert np.isfinite(u).all() and np.isfinite(v).all(),'nonfinite payload'
   assert np.array_equal(u[:,EXACT_COLUMNS],v[:,EXACT_COLUMNS]),'discrete chi events changed'
   diff=np.maximum(diff,np.max(np.abs(u-v),axis=0));scale=np.maximum(scale,np.max(np.abs(u),axis=0));count+=u.size
 error=diff/scale;tol=TOLERANCES[kind]
 result={'passed':bool(np.max(error)<=tol),'bitwise_equal':bits,'tolerance':tol,'max_normalized_error':float(np.max(error)),'column_normalized_error':error.tolist(),'column_max_absolute_error':diff.tolist(),'exact_chi_events':True,'scalar_values':count,'reference_payload_sha256':hd[0].hexdigest(),'candidate_payload_sha256':hd[1].hexdigest()}
 assert result['passed'],result
 return result

def history(reference,candidate,minimum_rows):
 reference,candidate=Path(reference),Path(candidate)
 x=np.loadtxt(reference,ndmin=2);y=np.loadtxt(candidate,ndmin=2);dt=.00046944203735617445
 old={round(r[0]/dt):r for r in x};new={round(r[0]/dt):r for r in y};cycles=sorted(old.keys()&new.keys())
 assert len(cycles)>=minimum_rows and x.shape[1]==y.shape[1]==12
 u=np.array([old[c] for c in cycles]);v=np.array([new[c] for c in cycles])
 assert np.isfinite(u).all() and np.isfinite(v).all() and np.array_equal(u[:,:2],v[:,:2])
 error=np.abs(u[:,2:]-v[:,2:])/np.maximum(np.abs(u[:,2:]),1e-30)
 result={'passed':bool(error.max()<=TOLERANCES['history_relative']),'matched_rows':len(cycles),'max_relative_error':float(error.max()),'relative_tolerance':TOLERANCES['history_relative'],'all_12_columns_bitwise_equal':bool(np.array_equal(u,v))}
 assert result['passed'],result
 return result

#!/usr/bin/env python3
"""Check read-only V12 profiles on equilibrium and a manufactured linear field."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for key in ['binary','checkpoint','coordinates','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);original=a.checkpoint.read_bytes();before=hashlib.sha256(original).hexdigest()

def run(cp,name,good=True):
 dest=out/(name+'.csv');r=subprocess.run([str(a.binary.resolve()),str(cp.resolve()),str(dest)],capture_output=True,text=True,timeout=120)
 (out/(name+'.log')).write_text(r.stdout+r.stderr)
 assert (r.returncode==0)==good,(name,r.stdout,r.stderr)
 return np.genfromtxt(dest,delimiter=',',names=True) if good else None
mink=run(a.checkpoint,'equilibrium')
assert len(mink)>1000 and len(set(zip(mink['domain'],mink['ray'],mink['r'])))==len(mink)
constant_error=max(np.max(abs(mink['v0']-1)),*[np.max(abs(mink['v'+str(f)])) for f in range(1,31)])
derivative_error=max(np.max(abs(mink['dr_v'+str(f)])) for f in range(31))
assert constant_error<1e-11 and derivative_error<1e-9
coords=np.genfromtxt(a.coordinates,delimiter=',',names=True);cells=int(np.frombuffer(original,dtype='<u8',count=1,offset=32)[0]);assert len(coords)==cells
fixture=bytearray(original);state=np.frombuffer(fixture,dtype='<f8',count=31*cells,offset=184).reshape(cells,31);state[:]=0
linear=coords['x']+2*coords['y']-3*coords['z'];state[:,0]=1+.001*linear;state[:,18]=.002*linear
for f,key in [(22,'x'),(23,'y'),(24,'z')]:state[:,f]=.01*coords[key]
with tempfile.TemporaryDirectory(dir=out) as temporary:
 cp=Path(temporary)/'manufactured-fixture.checkpoint';cp.write_bytes(fixture);data=run(cp,'linear')
 expected=data['x']+2*data['y']-3*data['z'];normal=expected/data['r'];metrics={}
 for domain,name in [(0,'cartesian'),(1,'spherical')]:
  mask=data['domain']==domain
  value=max(np.max(abs(data['v0'][mask]-(1+.001*expected[mask]))),np.max(abs(data['v18'][mask]-.002*expected[mask])))
  derivative=max(np.max(abs(data['dr_v0'][mask]-.001*normal[mask])),np.max(abs(data['dr_v18'][mask]-.002*normal[mask])))
  for f,key in [(22,'x'),(23,'y'),(24,'z')]:
   value=max(value,np.max(abs(data['v'+str(f)][mask]-.01*data[key][mask])))
   derivative=max(derivative,np.max(abs(data['dr_v'+str(f)][mask]-.01*data[key][mask]/data['r'][mask])))
  metrics[name]=dict(value_error=float(value),radial_derivative_error=float(derivative))
  assert value<(2e-13 if domain==0 else 5e-7) and derivative<(1e-10 if domain==0 else 5e-5),metrics
 for name,offset,payload in [('unadmitted',28,np.int32(0).tobytes()),('wrong_layers',156,np.int32(2).tobytes()),('bad_geometry',120,np.float64(.2).tobytes()),('nonfinite_compensation',184+31*cells*8,np.float64(float('nan')).tobytes())]:
  bad=bytearray(original);bad[offset:offset+len(payload)]=payload;cp.write_bytes(bad);run(cp,name,good=False)
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==before
result=dict(status='passed',checkpoint_sha256=before,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),rows=len(mink),
 equilibrium_value_error=float(constant_error),equilibrium_radial_derivative_error=float(derivative_error),manufactured=metrics,
 scope='Read-only interpolation/derivative audit on fixed rays; no evolution, characteristic reflection measurement or convergence qualification.')
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

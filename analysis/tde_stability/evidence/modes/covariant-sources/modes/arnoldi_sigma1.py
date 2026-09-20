from pathlib import Path
import sys,json,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parents[1]/'mode-analysis';sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE,profile,discrepancy
case='covariant_constant10';m=32
opts=('z4c/ccz4_covariant_sources=true','z4c/damp_lapse_scaled=true','z4c/damp_kappa1=1')
label=f'{case}-arnoldi{m}'
binary=ROOT.parent/'athena-covariant-modehook'
assert hashlib.sha256(binary.read_bytes()).hexdigest()=='ef1a5473e3bfb5722240cf96a65284a6a5a78c8987509d6de0719a572f84dad1'
op=Operator(str(ROOT/label),binary=binary,input_file=ROOT/'input.athinput',dt=.0375,overrides=opts)
seed=np.zeros(SHAPE);components=[]
for n in [0,1]:
 path=ROOT/f'covariant_constant03-arnoldi32-mode{n}-real.bin'
 v=np.fromfile(path).reshape(SHAPE);seed+=v/np.linalg.norm(v)
 components.append({'type':'sigma=.3 eigenvector','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
sys.path.insert(0,str(ROOT.parents[3]/'tst/regression'))
from z4c_background_restart import checkpoint
path=ROOT.parent/'covariant_const10_continuation/rst/rank_00000000/ks_background.00005.rst'
c=checkpoint(path)
assert len(c['state'])==1 and 500<c['time']<500.1
v=np.asarray(c['state'][0]).reshape(SHAPE)
assert np.isfinite(v).all() and np.linalg.norm(v)>0
seed+=v/np.linalg.norm(v)
components.append({'type':'finite sigma=1 checkpoint residual direction only','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'time_M':c['time'],'cycle':c['cycle'],'raw_max':float(np.max(abs(v)))})
(ROOT/f'{label}-seed.json').write_text(json.dumps({'components':components,'normalization':'Each full-state component has unit Euclidean norm before summing; mixture subsequently normalized. Checkpoint supplies a direction only, not an evolved background.'},indent=2)+'\n')
q=seed.ravel()/np.linalg.norm(seed);Q=np.empty((q.size,m+1));Q[:,0]=q;H=np.zeros((m+1,m));hist=[]
for k in range(m):
 start=time.monotonic();w=op.response(Q[:,k].reshape(SHAPE),80,1e-3).ravel()
 for _ in range(2):
  hh=Q[:,:k+1].T@w;H[:k+1,k]+=hh;w-=Q[:,:k+1]@hh
 H[k+1,k]=np.linalg.norm(w);Q[:,k+1]=w/H[k+1,k]
 ev,Y=np.linalg.eig(H[:k+1,:k+1]);idx=np.argsort(-abs(ev))[:6];row={'k':k+1,'seconds':time.monotonic()-start,'ritz':[]}
 for j in idx:
  mu=ev[j];res=abs(H[k+1,k]*Y[-1,j]);row['ritz'].append({'real':float(mu.real),'imag':float(mu.imag),'gamma':float(np.log(abs(mu))/3),'omega':float(np.angle(mu)/3),'relative_residual':float(res/abs(mu))})
 hist.append(row);(ROOT/f'{label}-progress.json').write_text(json.dumps(hist,indent=2)+'\n');print(json.dumps(row),flush=True)
np.savez_compressed(ROOT/f'{label}-krylov.npz',Q=Q,H=H)
results={'case':case,'steps':80,'dt':.0375,'map_interval_M':3,'krylov_dimension':m,'history':hist,'modes':[]}
for j in np.argsort(-abs(ev))[:4]:
 mu=ev[j]
 if mu.imag< -1e-10:continue
 v=(Q[:,:m]@Y[:,j]).reshape(SHAPE).astype(complex);v/=np.max(abs(v))
 vr=v.real;vi=v.imag
 rr=op.response(vr,80,1e-3,label=f'mode{len(results["modes"])}_realcheck')
 ri=op.response(vi,80,1e-3,label=f'mode{len(results["modes"])}_imagcheck') if np.max(abs(vi))>1e-10 else np.zeros_like(vr)
 resp=rr+1j*ri;expect=mu*v;dd=resp-expect
 stem=ROOT/f'{label}-mode{len(results["modes"])}';vr.tofile(str(stem)+'-real.bin');vi.tofile(str(stem)+'-imag.bin')
 rec={'real':float(mu.real),'imag':float(mu.imag),'gamma':float(np.log(abs(mu))/3),'omega':float(np.angle(mu)/3),'direct_relative_l2':float(np.linalg.norm(dd)/np.linalg.norm(expect)),'direct_active_relative_l2':float(np.linalg.norm(dd[ACTIVE])/np.linalg.norm(expect[ACTIVE])),'state_real_file':str(stem)+'-real.bin','state_imag_file':str(stem)+'-imag.bin','profile_real':profile(vr)}
 results['modes'].append(rec)
(ROOT/f'{label}-results.json').write_text(json.dumps(results,indent=2)+'\n')

from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parents[1]/'mode-analysis';sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE,profile,discrepancy
case=sys.argv[1];m=int(sys.argv[2]) if len(sys.argv)>2 else 32
opts={'covariant_alpha01':('z4c/ccz4_covariant_sources=true',),'covariant_constant01':('z4c/ccz4_covariant_sources=true','z4c/damp_lapse_scaled=true'),'covariant_constant03':('z4c/ccz4_covariant_sources=true','z4c/damp_lapse_scaled=true','z4c/damp_kappa1=.3')}[case]
label=f'{case}-arnoldi{m}'
op=Operator(str(ROOT/label),binary=ROOT.parent/'athena-covariant-modehook',input_file=ROOT/'input.athinput',dt=.0375,overrides=opts)
seed=np.zeros(SHAPE)
for n in [0,1]:
 v=np.fromfile(OLD/f'arnoldi_s40_m32_eps0.001-mode{n}.bin').reshape(SHAPE);seed+=v/np.linalg.norm(v)
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

from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parents[1]/'mode-analysis';sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE,profile,discrepancy
op=Operator(str(ROOT/'sigma1_mode_validation'),binary=ROOT.parent/'athena-covariant-modehook',input_file=ROOT/'input.athinput',dt=.0375,overrides=('z4c/ccz4_covariant_sources=true','z4c/damp_lapse_scaled=true','z4c/damp_kappa1=1'))
probe=Operator(str(ROOT/'sigma1_mode_physical_constraints'),binary=OLD/'athena-constraint-probe',input_file=OLD/'input-constraints.athinput')
def apply(v,steps,eps,label):
 a=op.response(v.real,steps,eps,label=label+'_real')
 if np.max(abs(v.imag))>1e-10:a=a+1j*op.response(v.imag,steps,eps,label=label+'_imag')
 return a

def constraints_real(v,label):
 cc=[]
 for sign in [1,-1]:
  lab=f'{label}_sign{sign}';probe.advance(sign*1e-3*v,0,label=lab)
  cc.append(np.fromfile(probe.directory/lab/'output.bin.constraints.bin').reshape(7,24,24,24))
 return (cc[0]-cc[1])/2e-3

def constraints(v,label):
 c=constraints_real(v.real,label+'_real')
 if np.max(abs(v.imag))>1e-10:c=c+1j*constraints_real(v.imag,label+'_imag')
 return c

def dx(v,axis):
 out=np.zeros((16,16,16),dtype=v.dtype)
 for o,c in [(-3,-1/60),(3,1/60),(-2,3/20),(2,-3/20),(-1,-3/4),(1,3/4)]:
  sl=[slice(4,20)]*3;sl[2-axis]=slice(4+o,20+o);out+=c*v[tuple(sl)]/.25
 return out

def Q(v):
 g=[[1,2,3],[2,4,5],[3,5,6]];tr=v[1]+v[4]+v[6]
 return np.stack([v[14+i,4:20,4:20,4:20]-sum(dx(v[g[i][j]],j) for j in range(3))+.5*dx(tr,i) for i in range(3)])

def relative(a,b):
 return {'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'active_relative_l2':float(np.linalg.norm((a-b)[ACTIVE])/np.linalg.norm(a[ACTIVE])),'max_abs':float(np.max(abs(a-b)))}

results=[]
for n,m in enumerate(json.loads((ROOT/'covariant_constant10-arnoldi32-results.json').read_text())['modes'][:2]):
 if m['direct_relative_l2']>1e-4 or m['direct_active_relative_l2']>1e-4:
  print('Skip unconverged Ritz candidate',n,m['direct_relative_l2'],m['direct_active_relative_l2'],flush=True);continue
 v=np.fromfile(m['state_real_file']).reshape(SHAPE)+1j*np.fromfile(m['state_imag_file']).reshape(SHAPE);mu=complex(m['real'],m['imag'])
 r=apply(v,80,1e-3,f'mode{n}_80');low=apply(v,80,3e-4,f'mode{n}_80low');r1=apply(v,1,1e-3,f'mode{n}_1')
 ray=np.vdot(v[ACTIVE],r1[ACTIVE])/np.vdot(v[ACTIVE],v[ACTIVE]);c=constraints(v,f'mode{n}_initial');cr=constraints(r,f'mode{n}_advanced');q=Q(v);qr=Q(r)
 rec={'index':n,'mu_real':mu.real,'mu_imag':mu.imag,'gamma':m['gamma'],'omega':m['omega'],'eigen_residual':relative(mu*v,r),'amplitude_convergence':relative(r,low),'one_step_eigen_residual':relative(mu**(1/80)*v,r1),'one_step_active_rayleigh_gamma':float(np.log(abs(ray))/.0375),'physical_constraints':{},'profile_real':profile(v.real)}
 for name,a,b in [('H',c[1,4:20,4:20,4:20],cr[1,4:20,4:20,4:20]),('M_cov',c[4:7,4:20,4:20,4:20],cr[4:7,4:20,4:20,4:20]),('Q_contrav',q,qr)]:
  idx=np.unravel_index(np.argmax(abs(a)),a.shape)
  rec['physical_constraints'][name]={'initial_rms':float(np.sqrt(np.mean(abs(a)**2))),'norm_gain':float(np.linalg.norm(b)/np.linalg.norm(a)),'relative_eigen_residual':float(np.linalg.norm(b-mu*a)/np.linalg.norm(b)),'initial_peak_xyz':[(z+.5)*.25-2 for z in idx[-3:][::-1]]}
 rec['one_step_active_rayleigh_omega']=float(np.angle(ray)/.0375)
 rec['one_step_active_rayleigh_mu_real']=float(ray.real)
 rec['one_step_active_rayleigh_mu_imag']=float(ray.imag)
 rec['e_fold_time_M']=1/rec['gamma'];rec['oscillation_period_M']=2*np.pi/abs(rec['omega']) if rec['omega'] else None
 rec['scope']='Approximate discrete complex mode independently checked; finite32-vector search is not a complete spectral proof. Faster real Ritz candidate remains unconverged.'
 results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='profile_real'},indent=2),flush=True)
 np.savez_compressed(ROOT/f'covariant10-mode{n}-physical-constraints.npz',H=c[1,4:20,4:20,4:20],M=c[4:7,4:20,4:20,4:20],Q=q,H_advanced=cr[1,4:20,4:20,4:20],M_advanced=cr[4:7,4:20,4:20,4:20],Q_advanced=qr)
 (ROOT/'sigma1-validated-modes.json').write_text(json.dumps(results,indent=2)+'\n')

(ROOT/'sigma1-validated-modes.json').write_text(json.dumps(results,indent=2)+'\n')

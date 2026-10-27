"""Locate sampled lapse/Gamma speed crossings and audit interpolated roots.

Root states use linear interpolation between adjacent saved physical states.
This diagnoses symbol degeneracy, not causation of a nonlinear failure.
"""
import argparse, hashlib, inspect, json
from pathlib import Path
import numpy as np
from puncture import PunctureEvolution
from gamma_driver import collar
from principal import symbol

p=argparse.ArgumentParser()
p.add_argument('--case',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
m=json.loads((a.case/'manifest.json').read_text())
assert m['arguments']['shift_gauge']=='moving-puncture' and m['status']!='running'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for f,h in m['checkpoint_hashes'].items():assert sha(a.case/f)==h
lib=a.case/'source'/Path(m['arguments']['library']).name
assert sha(lib)==m['library_sha256']
kwargs={k:v for k,v in m['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
e=PunctureEvolution(m['arguments']['n'],lib.resolve(),**kwargs)
z=np.load(a.case/'profiles.npz');rows=[];curves={}
times=list(z['t']);states=list(z['q'])
if not any(abs(t-float(z['last_time']))<1e-13 for t in times):
 times.append(float(z['last_time']));states.append(z['last'])
for i,(t,q) in enumerate(zip(times,states)):
 report=e.report(q.copy(),float(t));v=e.last;r=e.r
 alpha,chi,g=v['alpha'],v['chi'],v['g']
 F=alpha**2+4*e.O**2*alpha
 eta=1.0001*alpha/(alpha+1e-4*e.A)
 B=chi*(F+g*eta*.25*e.w**2)
 C=collar(r);use=(r>0)&(C<.5)
 needed=np.zeros(len(r));needed[use]=.75*(1-2*C[use])*B[use]/(1-C[use])
 gap=v['shift_lapse_gap_times_g'];j=1+np.argmin(gap[1:])
 roots=[]
 for k in np.flatnonzero(gap[1:-1]*gap[2:]<0)+1:
  def state(x):
   aa,cc,gg,ww=[np.interp(x,r[k:k+2],f[k:k+2]) for f in [alpha,chi,g,e.w]]
   O=(1-x*x)/2;A=(1+x*x)/2;ff=aa*aa+4*O*O*aa;ee=1.0001*aa/(aa+1e-4*A)
   bb=cc*(ff+gg*ee*.25*ww*ww);c=float(collar(x))
   mu=(1-c)*e.gamma_coefficient+c*.75*e.ratio*bb
   return aa,cc,gg,ff,ee,ww,mu
  def margin(x):
   aa,cc,gg,ff,ee,ww,mu=state(x)
   return 4*mu/3-cc*(ff+gg*ee*.25*ww*ww)
  # Recomputed interpolated coefficients can shift a bracket by roundoff.
  if margin(r[k])*margin(r[k+1])>=0:continue
  lo,hi=r[k],r[k+1]
  for _ in range(48):
   mid=(lo+hi)/2
   if margin(lo)*margin(mid)<=0:hi=mid
   else:lo=mid
  x=(lo+hi)/2
  aa,cc,gg,ff,ee,ww,mu=state(x)
  P=symbol(aa,cc,gg,ff,ee,ww,.005,.02,mu)
  speed=np.sqrt(4*mu/(3*gg));sv=np.linalg.svd(P-speed*np.eye(10),compute_uv=False)
  roots.append(dict(radius=x,relative_to_AH='exterior' if x>=report['apparent_horizon_r'] else 'interior',
                    common_speed=float(speed),geometric_multiplicity=int(sum(sv<1e-8)),
                    algebraic_multiplicity=2,two_smallest_singular_values=sv[-2:].tolist(),gap=float(margin(x))))
 exterior=r>=report['apparent_horizon_r'];diag=v['diag'][:,0];idx=np.flatnonzero(exterior)
 peak=idx[np.argmax(abs(diag[exterior]))]
 rows.append(dict(t=float(t),minimum_gap=float(gap[j]),minimum_gap_radius=float(r[j]),
                  largest_required_bulk_mu=float(max(needed)),roots=roots,
                  exterior_H_peak_radius=float(r[peak]),exterior_H_peak=float(abs(diag[peak])),
                  H_exterior_rms=report['H_exterior_rms'],pole=report['pole']))
 if i in [0,len(times)//2,len(times)-1]:
  curves[f't{i}']=np.c_[r,gap,B,diag,alpha,chi,g]
h=json.loads((a.case/'history.json').read_text())
first=next((x for x in h if x['shift_lapse_gap_times_g_min']<=0),None)
result=dict(case=str(a.case),manifest_sha256=sha(a.case/'manifest.json'),status=m['status'],stop_time=m['t'],
            first_nonpositive_report=None if first is None else {k:first[k] for k in ['t','H_exterior_rms','pole','shift_lapse_gap_times_g_min']},
            largest_required_bulk_mu_over_saved_profiles=max(x['largest_required_bulk_mu'] for x in rows),
            rows=rows,profile_columns=['r','gap_times_g','g_lapse_speed_squared','H','alpha','chi','g'],
            scope='Saved-time scalar-symbol diagnostic. Interpolated roots are estimates; no full-domain stability or causal attribution. Required coefficient is evaluated on the old trajectory, not a prediction of a changed trajectory.')
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(a.output/'profiles.npz',**curves)
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','profile_columns']},indent=2))
print(json.dumps(rows[-1],indent=2))

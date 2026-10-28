"""Boundary identities, native-kernel principal coefficients and degeneracies."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from boundary import normal_shift,normal_rate,vector_shift,vector_rate_identity
from principal import symbol,coefficient,speeds,eigenspace_audit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from kernel import Kernel

def native_symbol(k,alpha,chi,g):
 r=np.array([.43]);O=(1-r[0]**2)/2
 q=np.zeros((1,8));q[0,[0,1,6]]=[chi-1,g-1,alpha-(1+r[0]**2)/2]
 d=np.zeros_like(q);dd=d.copy();beta=np.zeros(1);db=beta.copy();ddb=beta.copy()
 def get():
  R,S,_=k.parts(r,q,d,dd,beta,db,ddb)
  return (R+S/O)[0]
 P=np.zeros((10,10));eps=1e-4
 for col,field in [(2,2),(3,3),(4,4)]:
  delta=eps*(O if col in [3,4] else 1)
  q[0,field]=delta;plus=get();q[0,field]=-delta;minus=get();q[0,field]=0
  P[[0,1,6],col]=((plus-minus)/(2*eps))[[0,1,6]]
 db[0]=eps;plus=get();db[0]=-eps;minus=get();db[0]=0
 P[[0,1,6],7]=((plus-minus)/(2*eps))[[0,1,6]]
 for col,field in [(0,0),(1,1),(6,6),(7,7),(3,3),(4,4),(5,5)]:
  result=[]
  for sign in [1,-1]:
   if col==7:ddb[0]=sign*eps
   elif col in [0,1,6]:dd[0,field]=sign*eps
   else:d[0,field]=sign*eps*(O if col in [3,4] else 1)
   result.append(get());d[:]=0;dd[:]=0;ddb[:]=0
  P[[2,3,4,5],col]=((result[0]-result[1])/(2*eps))[[2,3,4,5]]/[1,O,O,1]
 return P

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rng=np.random.default_rng(29);null=[];rates=[];vnull=[];vrates=[]
 for _ in range(100):
  alpha,chi,g=np.exp(rng.uniform(-5,3,3));at,ct,gt=rng.normal(size=3)*np.array([alpha,chi,g])
  beta=normal_shift(alpha,chi,g);null.append(abs((beta/alpha)**2/(chi/g)-1))
  dt=1e-5;fd=(normal_shift(alpha+dt*at,chi+dt*ct,g+dt*gt)-normal_shift(alpha-dt*at,chi-dt*ct,g-dt*gt))/(2*dt)
  rates.append(abs(fd-normal_rate(alpha,chi,g,at,ct,gt))/max(1,abs(beta)))
  m=rng.normal(size=(3,3));metric=m@m.T+np.eye(3);normal=rng.normal(size=3);trial=rng.normal(size=3)
  mt=rng.normal(size=(3,3));mt=(mt+mt.T)/2
  b=vector_shift(alpha,metric,normal,trial);inv=np.linalg.inv(metric);n=np.sqrt(normal@inv@normal)
  vnull.append(abs((b@normal/n)/alpha+1))
  bplus=vector_shift(alpha+dt*at,metric+dt*mt,normal,trial)
  bminus=vector_shift(alpha-dt*at,metric-dt*mt,normal,trial)
  vrates.append(abs((normal/n)@((bplus-bminus)/(2*dt))-vector_rate_identity(alpha,metric,normal,b,at,mt))/max(1,alpha))
 assert max(null)<2e-15 and max(vnull)<2e-13
 assert max(rates)<1e-8 and max(vrates)<1e-8
 k=Kernel(a.library);errors=[]
 for alpha,chi,g in [(.8,1.2,.9),(1.,1.,1.),(.2,.3,1.4)]:
  O=(1-.43**2)/2;F=alpha**2+4*O**2*alpha
  got=native_symbol(k,alpha,chi,g);expected=symbol(alpha,chi,g,F,0,0,.005,.02,0)
  expected[9]=0
  errors.append(float(np.max(abs(got-expected))))
 assert max(errors)<1e-8,errors
 from evolution import Evolution
 stage_checks=[]
 for amplitude in [.001,.01,-.01]:
  ev=Evolution(33,a.library,restoring=4.);state=np.zeros((33,10))
  state[:,0]=amplitude*ev.r**2;state[:,1]=.4*amplitude*ev.r**2
  state[:,6]=.3*amplitude*ev.r**2
  accepted=ev.step(state,1e-8)
  assert ev.stats['calls']==5 and ev.stats['max_relative_null']<2e-15
  al=ev.A[-1]+accepted[-1,6];ch=1+accepted[-1,0];gg=1+accepted[-1,1]
  assert abs((accepted[-1,7]-1)**2/al**2/(ch/gg)-1)<2e-15
  stage_checks.append(dict(amplitude=amplitude,stage_and_accepted_calls=ev.stats['calls'],max_relative_null=ev.stats['max_relative_null']))
 corners={}
 for ratio in [.5,1.,2.]:
  mu=.75*ratio;P=symbol(1,1,1,1,1,0,.005,.02,mu)
  cs=speeds(1,1,1,1,1,0,.005,.02,mu,beta=-1)
  corners[str(ratio)]=dict(audit=eigenspace_audit(P,1-cs),speeds=sorted(cs.tolist()),
   incoming=int(sum(cs< -1e-10)),outgoing=int(sum(cs>1e-10)),tangent=int(sum(abs(cs)<=1e-10)))
 assert not corners['1.0']['audit']['complete']
 assert corners['2.0']['audit']['complete'] and corners['2.0']['incoming']==1
 samples=[]
 for r in [.01,.2,.5,.79,.9,.999999]:
  for alpha,chi in [(1.,1.),(.01,1e-3),(1e-6,1e-10)]:
   g=1.2;F=alpha**2+(1-r*r)**2*alpha;eta=1.0001*alpha/(alpha+1e-4*(1+r*r)/2);w=float(r<.7)
   mu=coefficient(alpha,chi,g,F,eta,w,.005,.02)
   P=symbol(alpha,chi,g,F,eta,w,.005,.02,mu)
   eigen=np.linalg.eigvals(P);expected=-speeds(alpha,chi,g,F,eta,w,.005,.02,mu)
   error=float(np.max(abs(np.sort_complex(eigen)-np.sort_complex(expected))))
   samples.append(dict(r=r,alpha=alpha,chi=chi,speed_error=error,
     raw_eigenvector_condition=float(np.linalg.cond(np.linalg.eig(P)[1]))))
 assert max(x['speed_error'] for x in samples)<1e-7
 record=dict(boundary_tests=100,max_relative_null=max(null),max_vector_null=max(vnull),
   rate_fd_error=max(rates),vector_rate_fd_error=max(vrates),native_principal_errors=errors,
   scri_candidates=corners,collapsed_samples=samples,perturbed_RK_boundary_checks=stage_checks,
   limitations=['Numerical scalar-sector checks do not prove 3D strong hyperbolicity.',
   'Raw eigenvector conditioning depends on variables; no uniform puncture estimate.',
   'p/Omega and Theta/Omega reduction is interior-only; endpoint needs regularity.'])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:record[k] for k in ['boundary_tests','max_relative_null','max_vector_null','native_principal_errors']}))
if __name__=='__main__':main()

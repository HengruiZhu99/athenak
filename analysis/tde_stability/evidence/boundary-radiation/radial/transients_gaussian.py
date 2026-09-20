"""Finite-time tests of differential Bjørhus full-state operator; no PDE jobs."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.linalg import expm
from p_only import build,spectrum,PREV
sys.path.insert(0,str(PREV/'radial-full-boundary'));from summarize import offgrid
ROOT=Path(__file__).parent
TIMES=np.array([0.,1.,2.,5.,10.,20.,40.,80.,150.,300.])

def initial(r):
 n=len(r);R=1+r;a=r/R;ch=a*a;K=1/R**2;b=r/R**2
 x=r-2.;w=.45;f=np.exp(-x*x/(2*w*w));fp=-x/w**2*f;fpp=(x*x/w**4-1/w**2)*f
 states=np.zeros((8,n,3));states[6,:,0]=f;states[3,:,1]=f
 h=4*(fp-f/r)/3;hp=4*(fpp-fp/r+f/r**2)/3
 states[:,:,2]=np.array([f*2*r/R**3-2*ch*(fp+2*f/r)/3,h,-2*f/R**3,0*r,8*f/(3*R**3)-2*K*h/3,hp+3*h/r,f/R**2,f*(1-r)/R**3-b*fp])
 return states.reshape(8*n,3)

def run(n):
 start=time.monotonic();A,raw,r,D,C,rows,labels,replaced=build(n);N=len(r);U0=initial(r);a=r/(1+r);v=a*a-r/(1+r)**2
 names=['pure_lapse','Theta_pulse','constraint_free_radial_coordinate'];hist={name:[] for name in names}
 for t in TIMES:
  U=U0 if t==0 else expm(A*t)@U0
  for j,name in enumerate(names):
   state=U[:,j];der=A@state;th=C['Theta']@state;q=C['Q']@state;ham=C['H']@state;mom=C['M']@state
   rad=np.array([(C['Theta']@der)[0]+v[0]*((D@th)[0]+th[0]/(1+r[0])),(C['Q']@der)[0]+v[0]*((D@q)[0]+q[0]/r[0])])
   hist[name].append({'time_M':float(t),'state_L2_grid_over_initial':float(np.linalg.norm(state)/np.linalg.norm(U0[:,j])),'max_Theta':float(np.max(abs(th))),'max_Q':float(np.max(abs(q))),'max_H':float(np.max(abs(ham))),'max_M':float(np.max(abs(mom))),'constraint_L2_grid_over_initial_state':float(np.linalg.norm(np.r_[th,q,ham,mom])/np.linalg.norm(U0[:,j])),'actual_radiation_boundary_absolute':abs(rad).tolist(),'finite':bool(np.all(np.isfinite(state)))})
 o={'degree':n,'initial_profile':'Gaussian exp(-(r-2)^2/(2*0.45^2)); unit linear amplitude. Times use dense matrix exponential, no time-discretization error apart from floating point.','states':hist,'wall_seconds':time.monotonic()-start}
 (ROOT/f'transient-n{n}.json').write_text(json.dumps(o,indent=2)+'\n');print(n,'seconds',o['wall_seconds'],'final',[(k,v[-1]['constraint_L2_grid_over_initial_state']) for k,v in hist.items()],flush=True)
 return o
if __name__=='__main__':
 for n in [32,48,64,80]:
  o=spectrum(n);(ROOT/f'p-only-n{n}.json').write_text(json.dumps(o,indent=2)+'\n')
  dd={'degree':n,'inner_M':.2,'outer_M':4,'kappa1':.1,'near_constraint_only_branch':o['near_constraint_reference']};z=offgrid(dd);(ROOT/f'offgrid-n{n}.json').write_text(json.dumps(z,indent=2)+'\n')
  run(n)

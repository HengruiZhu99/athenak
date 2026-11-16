#!/usr/bin/env python3
"""Exact Schwarzschild coordinate fixtures, kept separate from native A evolution.

See docs/z4c_hyperboloidal/nonradiative-20261115 for chart admission limits.
Spacetime jets use forward AD independent of the extractor's finite differences.
"""
import argparse,importlib.util,json,sys,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vis/python'))
from scri_geometry import ricci,derivative_weights
from scri_frame import FrameTransport,generator_geometry
from scri_angular import spin_basis,weighted_projection
spec=importlib.util.spec_from_file_location('jet_fixture',Path(__file__).with_name('scri_geometry.py'))
jm=importlib.util.module_from_spec(spec);spec.loader.exec_module(jm)
J=jm.Jet

def log(x):return x.function(np.log(x.v),1/x.v,-1/x.v**2)
def outer(a,b):return [[a[i]*b[j] for j in range(4)] for i in range(4)]
def coords(t,x):return J(np.full(len(x),t),0),[J(x[:,i],i+1) for i in range(3)]
def profile(u,mass=1/3,amplitude=1,center=25,phase=0):
 z=u/mass;w=.5;tau=10
 envelope=(-((z-center)/tau)**2).exp()
 f=amplitude*mass*mass*(w*z+phase).sin()*envelope
 fp=amplitude*mass*envelope*(w*(w*z+phase).cos()-2*(z-center)/tau**2*(w*z+phase).sin())
 return f,fp

def physical(t,x,family,mass=1/3,parameter=.1):
 """Literal paper metrics at finite radius; signature -+++."""
 t,y=coords(t,x);jac=[[J(float(a==b)) for b in range(4)] for a in range(4)]
 if family=='bounce':
  angle=2*np.pi*t/(40*mass);a=2*mass*angle.sin()**4
  ap=8*np.pi/40*angle.sin()**3*angle.cos()
  # derivative: 2M * 4 * (2pi/40M) = 16pi/40
  ap=2*ap
  y[0]=y[0]+a;jac[1][0]=ap
 r=sum(v*v for v in y)**.5;n=[v/r for v in y]
 if family in ('schwarzschild','bounce'):
  k=[J(1),*n];g=outer(k,k)
  g=[[2*mass/r*g[a][b]+float(a==b)*(1 if a else -1) for b in range(4)] for a in range(4)]
 elif family=='gauge':
  f,fp=profile(t-r,mass,parameter)
  dv=[1+fp/r,*[v*(1-fp/r-f/r**2) for v in n]];dr=[J(0),*n]
  g=[[-(1-2*mass/r)*dv[a]*dv[b]+dv[a]*dr[b]+dr[a]*dv[b]
      +(float(a==b and a>0)-dr[a]*dr[b]) for b in range(4)] for a in range(4)]
 elif family=='rotation':
  # phi_old=phi_new+omega*u, right-handed about Cartesian z.
  w=parameter/mass;rot=[-w*y[1],w*y[0],J(0)]
  g=[[J(0) for _ in range(4)] for _ in range(4)];g[0][0]=-(1-2*mass/r)+sum(v*v for v in rot)
  for i in range(3):
   g[0][i+1]=g[i+1][0]=-n[i]+rot[i]
   for k in range(3):g[i+1][k+1]=float(i==k)-n[i]*n[k]
 else:raise ValueError(family)
 pulled=[[sum(jac[c][a]*g[c][d]*jac[d][b] for c in range(4) for d in range(4)) for b in range(4)] for a in range(4)]
 return jm.arrays(pulled,len(x))

def compact(t,x,family='schwarzschild',mass=1/3,parameter=.1,time_warp=0,conformal_warp=0,height=1):
 """Regular outgoing-null pullback with native Omega=(1-|x|^2)/2.

 R=r/Omega for the stationary/rotating cases. Retarded bounce translates
 the *outgoing-null* spatial chart by a(U), and is explicitly a counterpart,
 not the paper's oscillating Kerr-Schild-time chart. Gauge uses the paper's
 exact t-r argument including the Schwarzschild logarithm; its endpoint
 perturbation is flat (all derivatives zero) for the sine-Gaussian profile.
 """
 t,y=coords(t,x);r=sum(v*v for v in y)**.5;n=[v/r for v in y]
 om=(1-r*r)/2;dom=[J(0),*[-v for v in y]]
 q=height*(1-r)/(1+r);u=t+q;du=[J(1),*[-2*height*v/(1+r)**2 for v in n]]
 if time_warp:
  p2=(3*n[0]*n[0]-1)/2
  dp=[J(0),*[3*n[0]*(float(i==0)-n[0]*n[i])/r for i in range(3)]]
  du=[v*(1+time_warp*(u/mass).cos()*p2)+time_warp*mass*(u/mass).sin()*dp[a] for a,v in enumerate(du)]
  u=u+time_warp*mass*(u/mass).sin()*p2
 if family=='gauge':
  edge=np.max(abs(om.v))<2e-15
  if not edge:
   if np.any(abs(om.v)<2e-15):raise ValueError('split endpoint and interior gauge batches')
   R=r/om;dR=[J(0),*[(1+r*r)/(2*om*om)*v for v in n]]
   v=u+4*mass*log(R/(2*mass)-1)
   dv=[du[a]+4*mass*dR[a]/(R-2*mass) for a in range(4)]
   F,Fp=profile(v,mass,parameter)
   du=[du[a]+Fp*dv[a]/R-F*dR[a]/R**2 for a in range(4)]
   u=u+F/R
 # Z=Omega*X; evaluate Omega^2 dR = Omega d|Z|-|Z| dOmega.
 Z=y;dZ=[[J(float(a==i+1)) for a in range(4)] for i in range(3)]
 if conformal_warp:
  if family!='schwarzschild':raise ValueError('conformal radial chart is a separate control')
  p2=(3*n[0]*n[0]-1)/2
  dp=[J(0),*[3*n[0]*(float(i==0)-n[0]*n[i])/r for i in range(3)]]
  scale=(conformal_warp*(u/mass).sin()*p2).exp()
  dl=[conformal_warp*((u/mass).cos()*du[a]/mass*p2+(u/mass).sin()*dp[a]) for a in range(4)]
  Z=[v/scale for v in y]
  dZ=[[(dZ[i][a]-y[i]*dl[a])/scale for a in range(4)] for i in range(3)]
 if family=='retarded_bounce':
  angle=2*np.pi*u/(40*mass);a=2*mass*angle.sin()**4
  ap=16*np.pi/40*angle.sin()**3*angle.cos()
  Z=[y[0]+om*a,y[1],y[2]]
  dZ[0]=[dZ[0][k]+dom[k]*a+om*ap*du[k] for k in range(4)]
 L=sum(v*v for v in Z)**.5;N=[v/L for v in Z]
 dL=[sum(N[i]*dZ[i][a] for i in range(3)) for a in range(4)]
 angular=dZ
 if family=='rotation':
  w=parameter/mass;cross=[-Z[1],Z[0],J(0)]
  angular=[[dZ[i][a]+w*cross[i]*du[a] for a in range(4)] for i in range(3)]
 radial=[om*dL[a]-L*dom[a] for a in range(4)]
 f=1-2*mass*om/L
 g=[[-om*om*f*du[a]*du[b]-du[a]*radial[b]-radial[a]*du[b]
     +sum(angular[i][a]*angular[i][b] for i in range(3))-dL[a]*dL[b]
     for b in range(4)] for a in range(4)]
 return jm.arrays(g,len(x))

def grid(n=8):
 mu,w=np.polynomial.legendre.leggauss(n);phi=np.arange(2*n)*np.pi/n
 mu=np.repeat(mu,2*n);w=np.repeat(w,2*n)*np.pi/n;phi=np.tile(phi,n)
 return np.column_stack((mu,np.sqrt(1-mu*mu)*np.cos(phi),np.sqrt(1-mu*mu)*np.sin(phi))),w

def finite_checks():
 x=np.array([[6.,2.,1.],[-2.,7.,3.],[2.,-4.,8.]])/3
 report={}
 for family,parameter in [('schwarzschild',0),('rotation',.1),('rotation',.8),('bounce',2),('gauge',.01),('gauge',1),('gauge',10)]:
  g,dg,ddg=physical(8.3,x,family,parameter=parameter)
  R,_,_=ricci(g,dg,ddg)
  # Differentiate the AD metric values with independently generated polynomial
  # weights. This guards derivative/Jacobian wiring; vacuum Ricci is independent.
  error=0;h=2e-3
  for axis in range(4):
   vals=[]
   for k in range(-3,4):
    xx=x.copy();tt=8.3
    if axis==0:tt+=k*h
    else:xx[:,axis-1]+=k*h
    vals.append(physical(tt,xx,family,parameter=parameter)[0])
   fd=np.einsum('t,tqab->qab',derivative_weights(np.arange(-3,4)*h,0,1),vals)
   error=max(error,float(np.max(abs(fd-dg[:,axis]))))
  report[family+str(parameter)]={'ricci_max':float(np.max(abs(R))),'derivative_crosscheck':error}
  assert np.max(abs(R))<2e-11,(family,R)
  assert error<2e-9,(family,error)
 return report

def extract_case(family,parameter,output,duration=.5,dt=.01,lmax=4,time_warp=0,mass=1/3,conformal_warp=0,height=1):
 x,w=grid(max(8,lmax+2));times=np.linspace(0,duration,round(duration/dt)+1)
 transport=FrameTransport(x,w,lmax);data=[];flows=[];sources=[];nt=[];Tmax=[];null=[]
 for t in times:
  g,dg,ddg=compact(t,x,family,mass=mass,parameter=parameter,time_warp=time_warp,conformal_warp=conformal_warp,height=height)
  R,G,inv=ricci(g,dg,ddg);v,s,n,H=generator_geometry(x,inv,G)
  box=4*s*n;T=H-g*box[:,None,None]/4
  data.append((g,R,G));flows.append(v);sources.append(s);nt.append(n)
  Tmax.append(float(np.max(abs(T))));null.append(float(np.max(abs(np.einsum('qi,qi->q',v*n[:,None],-x)))))
 flows,sources,nt=map(np.array,(flows,sources,nt));states=transport.evolve(times,flows,sources,nt)
 rows=[];modes=[];nodal=[];pieces=[]
 for k in range(3,len(times)-3):
  idx=slice(k-3,k+4);d=derivative_weights(times[idx],times[k],1)@states[idx].reshape(7,-1)
  dd=derivative_weights(times[idx],times[k],2)@states[idx].reshape(7,-1)
  observed=transport.observe(states[k],*data[k],flows[k],sources[k],nt[k],dd.reshape(states[k].shape),d.reshape(states[k].shape))
  spin,labels=spin_basis(observed['direction'],lmax)
  project=weighted_projection(spin,observed['angular_weights']);coef=project@observed['news'];modes.append(coef);nodal.append(observed['news'])
  pieces.append([project@observed[key] for key in ('ricci_piece','hessian_piece','gradient_piece')])
  np.testing.assert_allclose(sum(observed[key] for key in ('ricci_piece','hessian_piece','gradient_piece')),observed['news'],atol=5e-14)
  row=observed['diagnostics'];row.update(time=float(times[k]),all_mode_max=float(np.max(abs(coef))),news_rms=float(np.sqrt(np.sum(w*abs(observed['news'])**2)/sum(w))),T_max=Tmax[k],null_max=null[k]);rows.append(row)
 output.mkdir(parents=True,exist_ok=False)
 np.savez(output/'cuts.npz',times=times[3:-3],modes=modes,nodal=nodal,pieces=pieces,ell_m=labels,frame=states,frame_times=times,x=x,weights=w)
 summary={'height':height,'conformal_warp':conformal_warp,'mass':mass,'family':family,'parameter':parameter,'time_warp':time_warp,'dt':dt,'lmax':lmax,'maxima':{key:max(abs(row[key]) for row in rows) for key in rows[0] if key!='time'},'native_A_evolution':False}
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'cuts.json').write_text(json.dumps(rows,indent=2)+'\n')
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 start=time.monotonic();report={'finite_physical':finite_checks(),'endpoint':[]}
 for name,par in [('schwarzschild',0),('rotation',.1),('rotation',.8),('retarded_bounce',2),('gauge',.01),('gauge',1),('gauge',10)]:
  r=extract_case(name,par,a.output/(name+str(par)),duration=.4,dt=.01)
  report['endpoint'].append(r);print(name,par,r['maxima'],flush=True)
 report['seconds']=time.monotonic()-start;report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

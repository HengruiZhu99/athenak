"""Isolated nonadvective experiment with an incoming characteristic boundary jet.

State 0:9 retains the original physical/chart variables and telegraph flux.
Column9 stores b+r. Only the endpoint entries of columns10:13 are used:
independent first-order reductions chi_r, g_r, beta_r at scri. Their rates are
endpoint derivatives of the total-field rates. The incoming gauge projection
changes these three jets and Lambda together, preserving frozen outgoing and
tangent amplitudes and the connection-constraint trace. Reduction defects are
reported. Independent tests reject this prototype: the augmented-jet constraint
is not the constraint reconstructed from the metric. It is not an admitted
outgoing-constraint-preserving boundary method or a well-posedness proof.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hyperbolic_shift_1d'))
from puncture import PunctureEvolution
from evolution import Evolution as RegularEvolution
from boundary import normal_rate
from constraint_damping import correction as damping_correction
from gauge import lapse_coefficient,collapsed_background_correction
from elliptic import connection
from numerics import puncture_dissipation
from nonadvective_gauge import coefficient,separation,shift_speeds,incoming_vector

class NonadvectiveEvolution:
 def __init__(self,n,library,puncture=True,reference_damping=1.,reference_rate=.1,
              gamma_coefficient=4.,ko=.5,lapse_floor=1e-10,cfl=.025,boundary_reduction_damping=0.,disabled_trace=None):
  if not np.isfinite([reference_damping,reference_rate,gamma_coefficient]).all() or min(reference_damping,reference_rate)<0 or gamma_coefficient<=0:raise ValueError('Invalid gauge parameters')
  self.boundary_reduction_damping=boundary_reduction_damping;self.disabled_trace=disabled_trace
  self.puncture=puncture;self.reference_damping=reference_damping;self.reference_rate=reference_rate;self.gamma_coefficient=gamma_coefficient
  common=dict(ko=ko,lapse_floor=lapse_floor,shift_gauge='moving-puncture',gamma_coefficient=gamma_coefficient,inner_z4_damping=8,restoring=4)
  self.base=PunctureEvolution(n,library,grid_stretch=0,shift_inner_stencil='odd',**common) if puncture else RegularEvolution(n,library,**common)
  self.stats=self.base.stats
  self.stats.update(min_stage_nonadvective_gap=np.inf,max_boundary_projection=0.,max_boundary_reduction_defect=0.,max_driver_boundary_defect=0.)
 def __getattr__(self,name):return getattr(self.base,name)

 def initial(self,amplitude=0.):
  q=np.zeros((len(self.r),13))
  if self.puncture:q[:,:10]=self.base.initial()
  else:
   z=(self.r-.4)/.12;mask=abs(z)<1;q[mask,6]=amplitude*np.exp(1-1/(1-z[mask]**2))
  q[:,9]=q[:,7] # b initially equals beta, including its analytic -r part.
  if self.puncture:
   chart=q[:,:8].copy();chart[1:,6]=np.log(q[1:,6]/self.chart.refs['alpha'][1:,0]);d,dd=self.op.derivatives(chart,np.ones(8))
   _,_,_,_,dp,_=self.chart.reconstruct(chart,d,dd)
   q[-1,10]=dp[-1,0];q[-1,11]=d[-1,1];q[-1,12]=-1+d[-1,7]
  else:
   d,_=self.op.derivatives(q[:,:8],self.parity[:8]);q[-1,10:13]=[d[-1,0],d[-1,1],-1+d[-1,7]]
  return self.materialize(q)

 def geometry(self,s,diagnostics=False):
  r=self.r;n=len(r)
  if self.puncture:
   chart=s[:,:8].copy();chart[1:,6]=np.log(s[1:,6]/self.chart.refs['alpha'][1:,0]);d,dd=self.op.derivatives(chart,np.ones(8))
   d[:,7],dd[:,7]=self.shift_op.derivatives(chart[:,7],-1)
   q,chi,alpha,khat,dp,ddp=self.chart.reconstruct(chart,d,dd)
   q[:,1]=s[:,1];dp[:,1]=d[:,1];ddp[:,1]=dd[:,1];q[:,7]=s[:,7];dp[:,7]=d[:,7];ddp[:,7]=dd[:,7]
  else:
   chart=s[:,:8];d,dd=self.op.derivatives(chart,self.parity[:8]);q=chart.copy();dp=d.copy();ddp=dd.copy();chi=1+s[:,0];alpha=self.A+s[:,6]
  beta=s[:,7]-r;g=1+s[:,1];db=-1+d[:,7];ddb=dd[:,7]
  reduction=np.array([s[-1,10]-dp[-1,0],s[-1,11]-dp[-1,1],s[-1,12]-db[-1]])
  dp[-1,0]=s[-1,10];dp[-1,1]=s[-1,11];db[-1]=s[-1,12]
  dp[-1,7]=db[-1]+1
  if self.puncture:
   native=q[1:].copy();native[:,0]=chi[1:];native[:,6]=alpha[1:];native[:,3]=khat[1:]
   R=np.zeros((n,8));S=R.copy();diag=np.zeros((n,12));R1=np.zeros((n-1,8));S1=R1.copy();diag1=np.zeros((n-1,12))
   arrays=[np.ascontiguousarray(z) for z in [native,dp[1:],ddp[1:],beta[1:],db[1:],ddb[1:]]]
   bad=self.function(n-1,np.ascontiguousarray(r[1:]),*arrays,4.,-.99,1,1.,R1,S1,diag1.ctypes.data if diagnostics else None)
   if bad:raise FloatingPointError('Native AF tensor failed at node '+str(bad))
   R[1:]=R1;S[1:]=S1;diag[1:]=diag1
   S[1:]+=damping_correction(r[1:],alpha[1:],q[1:,4],q[1:,5]-connection(r[1:],g[1:],dp[1:,1]),self.w[1:],8)
  else:
   R,S,diag=self.kernel.parts(r,q,dp,ddp,beta,db,ddb)
   S+=damping_correction(r,alpha,q[:,4],q[:,5]-connection(r,g,dp[:,1]),self.w,8)
  S[:,6]-=2*self.xi*alpha*q[:,6];S[:,6]+=collapsed_background_correction(r,q[:,6])
  dB,_=self.op.derivatives(s[:,8],np.ones(1));eta=np.zeros(n);positive=r>0
  eta[positive]=1.0001*alpha[positive]/(alpha[positive]+1e-4*self.A[positive])
  R[positive,6]+=eta[positive]*chi[positive]*(self.w[positive]*dB[positive]+(self.wp[positive]+2*self.w[positive]/r[positive])*s[positive,8])
  raw=np.zeros((n,8));start=int(self.puncture);raw[start:-1]=R[start:-1]+S[start:-1]/self.O[start:-1,None]
  ds=self.op.endpoint_first(S);raw[-1]=R[-1]-ds
  trace=np.zeros(8);trace[0]=chi[-1]-g[-1];trace[2]=q[-1,2]-dp[-1,1]/6+2*dp[-1,0]/3
  trace[3:5]=q[-1,3:5];trace[5]=q[-1,5]-connection(1.,g[-1],dp[-1,1])
  hs=self.h*(self.jac[-1] if self.puncture else 1.)
  penalty_trace=trace.copy()
  if self.disabled_trace is not None:penalty_trace[self.disabled_trace]=0.
  raw[-1]+=4/hs*S[-1]-8/hs*penalty_trace;raw[-1,6]=0.
  F=lapse_coefficient(r,alpha);cL2=chi*F/g+eta*chi*.25*self.w**2;mu=coefficient(r,g,beta,cL2,self.gamma_coefficient)
  bt=normal_rate(alpha[-1],chi[-1],g[-1],0.,raw[-1,0],raw[-1,1])
  defect=mu[-1]*s[-1,5]-self.reference_damping*(beta[-1]-(s[-1,9]-1))-bt
  return dict(q=q,chart=chart,d=d,dd=dd,dp=dp,chi=chi,alpha=alpha,g=g,beta=beta,db=db,raw=raw,R=R,S=S,diag=diag,trace=trace,raw_clock=float(R[-1,6]-ds[6]),mu=mu,cL2=cL2,bt=bt,defect=defect,reduction=reduction)

 def project(self,state,diagnostics=False):
  s=self.base.materialize(state);x=self.geometry(s)
  _,v=incoming_vector(x['chi'][-1],x['g'][-1],x['beta'][-1],x['mu'][-1])
  direction=np.zeros_like(s);direction[-1,[10,11,5,12]]=v[[0,1,5,7]]
  shifted=self.geometry(s+direction);slope=shifted['defect']-x['defect']
  if not np.isfinite(slope) or abs(slope)<1e-9:raise FloatingPointError('Singular incoming boundary compatibility')
  amplitude=-x['defect']/slope;s+=amplitude*direction
  final=self.geometry(s,diagnostics)
  scale=1+abs(final['mu'][-1]*s[-1,5])+abs(final['bt'])
  if abs(final['defect'])>1e-10*scale:raise FloatingPointError('Nonlinear boundary projection failed')
  self.stats['max_boundary_projection']=max(self.stats['max_boundary_projection'],abs(float(amplitude)))
  return s,final

 def materialize(self,state):return self.project(state)[0]

 def rhs(self,state,return_state=False,diagnostics=False):
  s,x=self.project(state,diagnostics);r=self.r;n=len(r);rate=np.zeros_like(s);raw=x['raw'];chi=x['chi'];alpha=x['alpha'];beta=x['beta']
  start=int(self.puncture)
  if self.puncture:
   rate[1:,0]=raw[1:,0]/chi[1:];rate[:,1]=raw[:,1];weight=self.chart.refs['w'][:,0]
   for f in [2,3,4]:rate[1:,f]=raw[1:,f]/weight[1:]
   rate[:,5]=raw[:,5];rate[1:,6]=raw[1:,6]
   adv=self.op.advection_derivative(x['chart'],np.ones(8),beta)
   correction=beta[:,None]*(adv-x['d']);correction[1:,6]*=alpha[1:];correction[-1]=0.
   rate[:,:7]+=correction[:,:7]
  else:
   rate[:,:7]=raw[:,:7];adv=self.op.advection_derivative(x['chart'],self.parity[:8],beta)
   correction=beta[:,None]*(adv-x['d']);correction[-1]=0.;rate[:,:7]+=correction[:,:7]
  ko=s[:,:10].copy()
  if self.puncture:ko[:,6]=x['chart'][:,6]
  kwargs=dict(measure=self.jac) if self.puncture else dict(include_first=True)
  diss=puncture_dissipation(ko,r,self.h,self.ko,.8,order=8,**kwargs)
  if self.puncture:diss/=self.jac[:,None];diss[1:,6]*=alpha[1:]
  rate[:,:7]+=diss[:,:7]
  rate[:,7]=x['mu']*s[:,5]-self.reference_damping*(s[:,7]-s[:,9])+diss[:,7]
  rate[:,9]=self.reference_rate*(s[:,7]-s[:,9])+diss[:,9]
  aB=self.op.advection_derivative(s[:,8,None],np.ones(1) if self.puncture else np.array([-1]),beta)[:,0]
  rate[:,8]=beta*aB+(.005*self.w*x['dp'][:,6]-s[:,8])/.02+diss[:,8]
  rate[-1,6]=0.;rate[-1,7]=x['bt']
  if self.puncture:
   b1=x['db'][0];b2=x['dd'][0,7]/2;g1=x['d'][0,1]
   rate[0,0]=2*b1;rate[0,1]=0.;rate[0,6]=2*b1-6+2*self.xi[0]
   rate[0,2]=2*b1*s[0,2];rate[0,3]=2*b1*(s[0,3]-3)+8*1.99*s[0,4]
   rate[0,4]=2*b1*s[0,4]-8*1.01*s[0,4]
   rate[0,5]=2*b1*s[0,5]-4*b1*g1+16*b2/3-8*(s[0,5]-4*g1)
   rate[0,7]=0.;rate[0,8]=-s[0,8]/.02;rate[0,9]=0.
  physical_rate=rate[:,:9].copy()
  if self.puncture:
   physical_rate[:,0]*=chi
   for f in [2,3,4]:physical_rate[:,f]*=weight
  rate[-1,10:13]=[self.op.endpoint_first(physical_rate[:,0]),self.op.endpoint_first(rate[:,1]),self.op.endpoint_first(rate[:,7])]
  hs=self.h*(self.jac[-1] if self.puncture else 1.)
  rate[-1,10:13]-=self.boundary_reduction_damping/hs*x['reduction']
  if not np.isfinite(rate).all():self.failure_state=s.copy();raise FloatingPointError('Nonfinite nonadvective RHS')
  gap=separation(beta,x['g'],x['mu'],x['cL2']);i=start+int(np.argmin(gap[start:]))
  self.stats['min_stage_nonadvective_gap']=min(self.stats['min_stage_nonadvective_gap'],float(gap[i]))
  if gap[i]<=0:self.failure_state=s.copy();raise FloatingPointError(f'Nonadvective speed separation failed at r={r[i]}')
  vm,vp=shift_speeds(beta,x['g'],x['mu']);light=alpha*np.sqrt(chi/x['g']);cL=np.sqrt(x['cL2'])
  speed=np.maximum.reduce([abs(vm),abs(vp),abs(beta)+cL,abs(beta)+light]);speed=speed/self.jac if self.puncture else speed
  null=chi[-1]/x['g'][-1]-(beta[-1]/alpha[-1])**2
  st=self.stats;st['calls']+=1
  for key,value in [('max_null',abs(null)),('max_relative_null',abs(null)/(chi[-1]/x['g'][-1])),('max_driver_boundary_defect',abs(x['defect'])),('max_pole',max(abs(x['S'][-1]))),('max_trace',max(abs(x['trace']))),('max_speed',max(speed[start:])),('max_boundary_reduction_defect',max(abs(x['reduction'])))]:st[key]=max(st.get(key,0),float(value))
  x.update(rate=rate,physical_rate=physical_rate,gap=gap,shift_minus=vm,shift_plus=vp)
  self.last=x
  return (rate,s) if return_state else rate

 def step(self,q,dt):
  a=self.rhs(q);b=self.rhs(q+dt*a/2);c=self.rhs(q+dt*b/2);d=self.rhs(q+dt*c)
  inc=dt*(a+2*b+2*c+d)/6
  if 'compensation' not in self.__dict__:self.compensation=np.zeros_like(q)
  adjusted=inc-self.compensation;accepted=q+adjusted;self.compensation=(accepted-q)-adjusted
  _,accepted=self.rhs(accepted,return_state=True);self.compensation[-1,[5,6,7,10,11,12]]=0.
  return accepted

 def report(self,q,t):
  self.rhs(q,diagnostics=True);x=self.last;weight=self.r**2*(self.jac if self.puncture else 1);weight[[0,-1]]*=.5;start=int(self.puncture)
  horizon={}
  if self.puncture:
   from horizon import apparent_horizons
   horizon=apparent_horizons(self.r[1:-1],x['chi'][1:-1]*np.sqrt(x['g'][1:-1]),x['diag'][1:-1,11])
  exterior=(self.r>horizon.get('apparent_horizon_r',0))&(self.r>0);norms={}
  for i,k in enumerate(['H','M','Theta','Z']):
   norms[k+'_exterior_rms']=float(np.sqrt(np.sum(weight[exterior]*x['diag'][exterior,i]**2)/sum(weight[exterior])))
   interior=(~exterior)&(self.r>0);norms[k+'_interior_max']=float(max(abs(x['diag'][interior,i]))) if any(interior) else 0.
  speedfields=x['physical_rate']
  return dict(t=t,**horizon,**norms,**self.stats,pole=float(max(abs(x['S'][-1]))),trace=float(max(abs(x['trace']))),scri_poles=x['S'][-1].tolist(),scri_traces=x['trace'].tolist(),raw_clock_rate=x['raw_clock'],driver_boundary_residual=float(x['defect']),boundary_reduction_defect=x['reduction'].tolist(),alpha_min=float(min(x['alpha'][start:])),chi_min=float(min(x['chi'][start:])),g_min=float(min(x['g'])),beta_min=float(min(x['beta'])),beta_max=float(max(x['beta'])),reference_gap_max=float(max(abs(q[start:,7]-q[start:,9]))),nonadvective_gap_min=float(min(x['gap'][start:])),exterior_reference_gap_rms=float(np.sqrt(np.sum(weight[exterior]*(q[exterior,7]-q[exterior,9])**2)/sum(weight[exterior]))),reference_rate_max=float(max(abs(x['rate'][start:,9]))),exterior_beta_rate_rms=float(np.sqrt(np.sum(weight[exterior]*x['rate'][exterior,7]**2)/sum(weight[exterior]))),exterior_metric_rate_rms=float(np.sqrt(np.sum(weight[exterior]*x['rate'][exterior,1]**2)/sum(weight[exterior]))),exterior_field_rate_rms=float(np.sqrt(np.sum(weight[exterior,None]*speedfields[exterior]**2)/sum(weight[exterior]))),exterior_Lambda_rms=float(np.sqrt(np.sum(weight[exterior]*q[exterior,5]**2)/sum(weight[exterior]))),floor_active_nodes=int(sum(x['alpha'][start:-1]<=self.lapse_floor*self.O[start:-1]*(1+1e-12))))

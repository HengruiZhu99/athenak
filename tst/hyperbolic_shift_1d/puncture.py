"""Live-metric AF chart: initial data and positive-node stencils from baseline.

Actual total lapse is the RK variable away from the coefficient endpoint.
All positive-radius nodes evolve. The r=0 slot contains asymptotic coefficients.
"""
import ctypes,sys
from pathlib import Path
import numpy as np
from evolution import Evolution
from boundary import normal_shift,normal_rate
from principal import coefficient
from shift_source import background
from metric_driver import transport_rate
from gamma_driver import parameters as gamma_parameters
from puncture_chart import PunctureChart
from telegraph import options,window
from wormhole_isotropic import IsotropicWormhole
from gauge import lapse_coefficient,collapsed_background_correction
from numerics import puncture_dissipation,RadialOperators
from elliptic import connection,solve

class PunctureEvolution(Evolution):
 def __init__(self,n,library,ratio=2.,damping=1.,ko=.5,restoring=4.,inner_shift=0.,lapse_floor=1e-10,shift_source='reference',connection_driver='evolved',shift_gauge='separated',gamma_coefficient=1.,grid_stretch=16.,shift_inner_stencil='positive'):
  super().__init__(n,library,ratio,damping,ko,restoring=restoring,inner_shift=inner_shift,lapse_floor=lapse_floor,shift_source=shift_source,connection_driver=connection_driver,shift_gauge=shift_gauge,gamma_coefficient=gamma_coefficient)
  opts=options(n,library);opts.grid_map='inner-exp-scri';opts.grid_stretch=grid_stretch
  opts.puncture_stencil='positive';opts.mapped_ko='local';opts.ko_order=8
  self.chart=PunctureChart(opts);self.r=self.chart.r;self.h=self.chart.h;self.op=self.chart.op
  if shift_inner_stencil not in ['positive','odd']:raise ValueError('Unknown shift inner stencil')
  if shift_inner_stencil=='odd' and shift_gauge!='moving-puncture':raise ValueError('Odd shift stencil is qualified only for the unsuppressed driver')
  self.shift_inner_stencil=shift_inner_stencil
  # A regular spherical gauge vector is distinct from the AF geometry chart.
  self.shift_op=RadialOperators(self.r,width=7,subtract_local=True) if shift_inner_stencil=='odd' else self.op
  self.O=self.chart.omega;self.A=(1+self.r**2)/2;self.jac=self.chart.grid_jacobian
  self.w,self.wp=window(self.r,width=.24,inner_profile='af-regular')
  self.xi=self.chart.restoring_strength
  function=self.kernel.lib.spherical_parts_positive_totals
  function.argtypes=self.kernel.lib.spherical_parts.argtypes;function.restype=ctypes.c_int
  # The tensor adapter already supports a null diagnostic pointer. Use a
  # separate ctypes callable, preserving the legacy ndarray-based signature.
  prototype=ctypes.CFUNCTYPE(ctypes.c_int,*function.argtypes[:-1],ctypes.c_void_p)
  self.function=prototype(('spherical_parts_positive_totals',self.kernel.lib))

 def initial(self):
  initial=IsotropicWormhole(1/3,96);state,iq,d,dd,_=self.chart.initial_wormhole(initial)
  q=np.zeros((len(self.r),10));q[:,:8]=state
  _,chi,alpha,_=self.chart.reconstruct(state)
  q[1:,6]=alpha[1:]
  # Match the initial spatial coordinates and elliptic initial shift. Subsequent
  # stages use only the local driver. No elliptic solve is used during evolution.
  integrand=np.zeros(len(q));integrand[1:]=1.5*alpha[1:]*iq[1:,2]/self.r[1:]
  beta=self.r*(-1-self.op.tail_integral(integrand))
  q[:,7]=beta+self.r
  q[0,7]=0.
  return self.materialize(q)

 def materialize(self,state):
  q=state.copy()
  if self.shift_gauge=='moving-puncture':q[0,9]=0. # dependent regular radial-driver trace at the spherical puncture
  active=(q[1:-1,6]<self.lapse_floor*self.O[1:-1])
  if np.any(active):
   indices=np.flatnonzero(active)+1;q[indices,6]=self.lapse_floor*self.O[indices]
   self.stats['floor_calls']+=1;self.stats['max_floor_radius']=max(self.stats['max_floor_radius'],float(max(self.r[indices])))
  if not np.isfinite(q).all() or np.any(q[1:,6]<=0) or np.any(1+q[:,1]<=0):
   self.failure_state=q.copy()
   bad=np.argwhere(~np.isfinite(q))
   if len(bad):i,f=bad[0];reason=f'nonfinite field {f}'
   elif np.any(1+q[:,1]<=0):i=int(np.argmin(1+q[:,1]));reason=f'g={1+q[i,1]:.17g}'
   else:i=int(np.argmin(q[1:,6]))+1;reason=f'alpha={q[i,6]:.17g}'
   raise FloatingPointError(f'Invalid AF stage: node {i}, r={self.r[i]:.17g}, {reason}')
  stored=q[:,:8].copy();stored[1:,6]=np.log(q[1:,6]/self.chart.refs['alpha'][1:,0])
  _,chi,alpha,_=self.chart.reconstruct(stored)
  q[-1,7]=1+normal_shift(alpha[-1],chi[-1],1+q[-1,1]);q[0,7]=0.
  return q

 def rhs(self,state,return_state=False,diagnostics=False):
  s=self.materialize(state);r=self.r;n=len(r);physical=slice(1,None)
  chart=s[:,:8].copy();chart[1:,6]=np.log(s[1:,6]/self.chart.refs['alpha'][1:,0])
  d,dd=self.op.derivatives(chart,np.ones(8))
  if self.shift_inner_stencil=='odd':d[:,7],dd[:,7]=self.shift_op.derivatives(chart[:,7],-1)
  q,chi,alpha,khat,dp,ddp=self.chart.reconstruct(chart,d,dd)
  q[:,1]=s[:,1];dp[:,1]=d[:,1];ddp[:,1]=dd[:,1]
  q[:,7]=s[:,7];dp[:,7]=d[:,7];ddp[:,7]=dd[:,7]
  beta=-r+s[:,7];db=-1+d[:,7];ddb=dd[:,7];g=1+s[:,1]
  native=q[1:].copy();native[:,0]=chi[1:];native[:,6]=alpha[1:];native[:,3]=khat[1:]
  R=np.zeros((n,8));S=R.copy();diag=np.zeros((n,12))
  rr=np.ascontiguousarray(r[1:]);R1=np.zeros((n-1,8));S1=R1.copy();diag1=np.zeros((n-1,12))
  arrays=[np.ascontiguousarray(z) for z in [native,dp[1:],ddp[1:],beta[1:],db[1:],ddb[1:]]]
  bad=self.function(n-1,rr,*arrays,4.,-.99,1,1.,R1,S1,diag1.ctypes.data if diagnostics else None)
  if bad:raise FloatingPointError(f'Native AF tensor failed at node {bad}')
  R[1:]=R1;S[1:]=S1;diag[1:]=diag1
  S[:,6]-=2*self.xi*alpha*q[:,6];S[:,6]+=collapsed_background_correction(r,q[:,6])
  dB,_=self.op.derivatives(s[:,8],np.ones(1));eta=np.zeros(n)
  eta[1:]=1.0001*alpha[1:]/(alpha[1:]+1e-4*self.A[1:])
  R[1:,6]+=eta[1:]*chi[1:]*(self.w[1:]*dB[1:]+(self.wp[1:]+2*self.w[1:]/r[1:])*s[1:,8])
  raw=np.zeros((n,8));raw[1:-1]=R[1:-1]+S[1:-1]/self.O[1:-1,None]
  raw[-1]=R[-1]-self.op.endpoint_first(S)
  trace=np.zeros(8);trace[0]=chi[-1]-g[-1];trace[2]=q[-1,2]-dp[-1,1]/6+2*dp[-1,0]/3
  trace[3:5]=q[-1,3:5];trace[5]=q[-1,5]-connection(1.,g[-1],dp[-1,1])
  hs=self.h*self.jac[-1];raw[-1]+=4/hs*S[-1]-8/hs*trace;raw[-1,6]=0.
  F=lapse_coefficient(r,alpha);mu=coefficient(alpha,chi,g,F,eta,self.w,.005,.02,self.ratio,inner_shift=self.inner_shift,omega=self.O)
  xi_shift=np.full(n,self.restoring)
  if self.shift_gauge=='moving-puncture':mu,xi_shift=gamma_parameters(r,mu,self.restoring,self.gamma_coefficient)
  rate=np.zeros_like(s);rate[1:,0]=raw[1:,0]/chi[1:];rate[:,1]=raw[:,1]
  weight=self.chart.refs['w'][:,0]
  for f in [2,3,4]:rate[1:,f]=raw[1:,f]/weight[1:]
  rate[:,5]=raw[:,5];rate[1:,6]=raw[1:,6]
  # Transport in the same positive-node reference variables as the baseline.
  adv=self.op.advection_derivative(chart,np.ones(8),beta)
  if self.shift_inner_stencil=='odd':adv[:,7]=self.shift_op.advection_derivative(chart[:,7,None],np.array([-1]),beta)[:,0]
  correction=beta[:,None]*(adv-d)
  correction[1:,6]*=alpha[1:];rate[:,:7]+=correction[:,:7]
  ko_state=s.copy();ko_state[:,6]=chart[:,6]
  diss=puncture_dissipation(ko_state,r,self.h,self.ko,.8,measure=self.jac,order=8)/self.jac[:,None]
  diss[1:,6]*=alpha[1:];rate[:,:7]+=diss[:,:7]
  db_driver=-1+adv[:,7]
  source,target_correction,target_prime=background(r,alpha,q[:,6],dp[:,6],self.shift_source,self.w,self.wp)
  rate[:,7]=beta*db_driver+source+mu*s[:,9]
  rate[:-1,7]-=xi_shift[:-1]*(s[:-1,7]+target_correction[:-1])/self.O[:-1];rate[:,7]+=diss[:,7]
  aB=self.op.advection_derivative(s[:,8,None],np.ones(1),beta)[:,0]
  rate[:,8]=beta*aB+(.005*self.w*dp[:,6]-s[:,8])/.02+diss[:,8]
  rate[-1,6]=0.
  bt=normal_rate(alpha[-1],chi[-1],g[-1],0.,chi[-1]*rate[-1,0],rate[-1,1])
  source_limit=xi_shift[-1]*(d[-1,7]+target_prime[-1])
  s[-1,9]=(bt-beta[-1]*db_driver[-1]-source[-1]-source_limit)/mu[-1]
  aD=self.shift_op.advection_derivative(s[:,9,None],np.array([-1]) if self.shift_inner_stencil=='odd' else np.ones(1),beta)[:,0]
  drive=rate[:,5]-beta*adv[:,5]
  if self.connection_driver=='metric':
   drive[1:]=transport_rate(r[1:],g[1:],dp[1:,1],alpha[1:],r[1:]+dp[1:,6],q[1:,2],dp[1:,2],beta[1:],db[1:],ddb[1:])
  rate[:,9]=beta*aD+drive-self.damping*s[:,9]+diss[:,9]
  rate[-1,7]=bt;rate[-1,9]=0.
  # Leading AF coefficient equations, rederived with g=1+g1*r+..., beta=b1*r+b2*r^2.
  b1=db[0];b2=ddb[0]/2;g1=d[0,1]
  rate[0,0]=2*b1;rate[0,1]=0.;rate[0,6]=2*b1-6+2*self.xi[0]
  rate[0,2]=2*b1*s[0,2];rate[0,3]=2*b1*(s[0,3]-3);rate[0,4]=2*b1*s[0,4]
  rate[0,5]=2*b1*s[0,5]-4*b1*g1+16*b2/3
  rate[0,7]=0.;rate[0,8]=-s[0,8]/.02;rate[0,9]=rate[0,5]-self.damping*s[0,9]
  if self.connection_driver=='metric':rate[0,9]=4*b1*g1+16*b2/3-self.damping*s[0,9]
  if self.shift_gauge=='moving-puncture':rate[0,9]=0. # beta(0)=0 implies D(0)=0, not a second independent datum
  if not np.isfinite(rate).all():raise FloatingPointError('Nonfinite AF RHS')
  light=np.sqrt(chi[1:]*F[1:]/g[1:]+eta[1:]*chi[1:]*.005*self.w[1:]**2/.02)
  shift=np.sqrt(4*mu[1:]/(3*g[1:]));speed=np.max((abs(beta[1:])+np.maximum(light,shift))/self.jac[1:])
  null=chi[-1]/g[-1]-(beta[-1]/alpha[-1])**2;st=self.stats;st['calls']+=1
  for key,value in [('max_null',abs(null)),('max_relative_null',abs(null)/(chi[-1]/g[-1])),
                    ('max_pole',max(abs(S[-1]))),('max_trace',max(abs(trace))),('max_speed',speed),
                    ('max_shift_pole',abs(self.restoring*(s[-1,7]+target_correction[-1]))),
                    ('max_driver_boundary_defect',abs(bt-(beta[-1]*db_driver[-1]+source[-1]+mu[-1]*s[-1,9]+source_limit)))]:
   st[key]=max(st.get(key,0),float(value))
  used=rate[1:-1,:7].copy();used[:,0]*=chi[1:-1]
  for field in [2,3,4]:used[:,field]*=weight[1:-1]
  self.last=dict(diag=diag,alpha=alpha,chi=chi,g=g,beta=beta,pole=S[-1],trace=trace,
                  light_speed=np.r_[0.,alpha[1:]*np.sqrt(chi[1:]/g[1:])],lapse_speed=np.r_[0.,light],shift_speed=np.r_[0.,shift],
                  shift_mu=mu,shift_restoring=xi_shift,
                  shift_lapse_gap_times_g=4*mu/3-chi*(F+g*eta*.25*self.w**2),
                  physical=q,physical_first=dp,raw_clock_rate=float(R[-1,6]-self.op.endpoint_first(S)[6]),
                  rate_difference=used-raw[1:-1,:7],original_equation_rate_difference=float(np.max(abs(used-raw[1:-1,:7]))))
  return (rate,s) if return_state else rate

 def report(self,q,t):
  self.rhs(q,diagnostics=True);x=self.last
  from horizon import apparent_horizons
  # b=g^(-1/2), so chi*sqrt(g) is the factor in the areal radius.
  horizon=apparent_horizons(self.r[1:-1],x['chi'][1:-1]*np.sqrt(x['g'][1:-1]),x['diag'][1:-1,11])
  weight=self.r**2*self.jac;weight[[0,-1]]*=.5
  exterior=self.r>horizon.get('apparent_horizon_r',np.inf)
  norms={}
  for i,key in enumerate(['H','M','Theta','Z']):
   norms[key+'_interior_max']=float(max(abs(x['diag'][1:][~exterior[1:],i]))) if np.any(~exterior[1:]) else 0.
   norms[key+'_exterior_rms']=float(np.sqrt(sum(weight[exterior]*x['diag'][exterior,i]**2)/sum(weight[exterior]))) if np.any(exterior) else None
  return dict(t=t,**horizon,**norms,**self.stats,pole=float(max(abs(x['pole']))),trace=float(max(abs(x['trace']))),
              shift_lapse_gap_times_g_min=float(min(x['shift_lapse_gap_times_g'][1:])),
              beta_inner=float(x['beta'][1]),beta_min=float(min(x['beta'])),beta_max=float(max(x['beta'])),
              inner_gauge_outgoing_max=float(max(-x['beta'][1]+x['lapse_speed'][1],-x['beta'][1]+x['shift_speed'][1])),
              alpha_min=float(min(x['alpha'][1:])),g_min=float(min(x['g'])),metric_change=float(max(abs(q[:,1]))),
              raw_clock_rate=x['raw_clock_rate'],scri_metric=float(x['g'][-1]),
              scri_poles=x['pole'].tolist(),scri_traces=x['trace'].tolist(),
              floor_active_nodes=int(np.count_nonzero(x['alpha'][1:-1]<=self.lapse_floor*self.O[1:-1]*(1+1e-12))),
              original_equation_rate_difference=x['original_equation_rate_difference'])

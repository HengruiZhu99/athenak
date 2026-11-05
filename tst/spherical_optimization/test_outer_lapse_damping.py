#!/usr/bin/env python3
"""Shared lapse damping: default parity, reference balance, core and scri jets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from kernel import Kernel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--baseline',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 old=Kernel(a.baseline,separated=True,compact_end=.97);base=Kernel(a.library,separated=True,compact_end=.97);new=Kernel(a.library,separated=True,compact_end=.97,outer_lapse_damping=.25)
 r=np.linspace(.001,.999,257);initial,src,om=base.initial(r);changed,source,omega=new.initial(r)
 for x,y in zip(old.initial(r),base.initial(r)):assert np.array_equal(x,y)
 assert np.array_equal(initial,changed) and np.array_equal(om,omega)
 rng=np.random.default_rng(2601);q=initial.copy();q[:,[3,4,5,6,7,8,9,10,11]]+=rng.normal(0,.02,(len(r),9));d=rng.normal(0,.02,q.shape);dd=rng.normal(0,.03,q.shape)
 before=base.parts(r,q,d,dd,src);after=new.parts(r,q,d,dd,source)
 for x,y in zip(old.parts(r,q,d,dd,src),before):assert np.array_equal(x,y)
 # Independent source-balanced identity: delta lapse_t = -2 delta_xi W alpha (alpha-alpha_*)/Omega.
 begin,end=base.config[7:9];t=np.clip((r-begin)/(end-begin),0,1);W=t**5*(126+t*(-420+t*(540+t*(-315+70*t))))
 expected=-2*(.25-1)*W*q[:,8]*(q[:,8]-src[:,0])/om[:,0]
 actual=after[0][:,8]-before[0][:,8]+(after[1][:,8]-before[1][:,8])/om[:,0]
 error=float(np.max(abs(actual-expected)));assert error<2e-10,error
 core=r<=begin
 for x,y in zip(before,after):assert np.array_equal(x[core],y[core])
 other=np.arange(12)!=8
 for x,y in zip(before[:2],after[:2]):assert np.array_equal(x[:,other],y[:,other])
 # Initial reference gauge must stay balanced although its prescribed sources change.
 p=base.parts(r,initial,d,dd,src);P=new.parts(r,initial,d,dd,source)
 stationary=float(np.max(abs(P[0][:,8]-p[0][:,8]+(P[1][:,8]-p[1][:,8])/om[:,0])));assert stationary<2e-11
 symbol_error=float(np.max(abs(base.symbol(.9,q[-30])-new.symbol(.9,q[-30]))));assert symbol_error<2e-10,symbol_error
 # Actual shared bulk pole versus analytic scri derivative, arbitrary nonstationary jets.
 Q=np.array([1.08,1.04,1/np.sqrt(1.04),.02,-.01,-3.01,.001,.03,.93,-1.02,.02,.13]);D=.003*np.arange(1,13);DD=-.001*np.arange(1,13)
 assert np.array_equal(old.analytic_normal(Q,D,DD),base.analytic_normal(Q,D,DD))
 rows=[]
 for k in (base,new):
  normal=k.analytic_normal(Q,D,DD);errors=[]
  for step in (2e-4,1e-4,5e-5):
   dx=-step*np.arange(5);nodes=np.ascontiguousarray(1+dx);values=Q+dx[:,None]*D+.5*dx[:,None]**2*DD;first=D+dx[:,None]*DD
   _,s,_=k.initial(nodes);_,poles,_,_=k.parts(nodes,values,first,np.tile(DD,(5,1)),s)
   fd=np.array([25,-48,36,-16,3])@poles[:,8]/(12*step);errors.append(float(abs(fd-normal[8])))
  assert max(errors)<1e-9,errors
  rows.append(dict(damping=k.outer_lapse_damping,lapse_normal=float(normal[8]),finite_difference_errors=errors))
 for bad in (0.,-1.,float('nan'),float('inf')):
  try:Kernel(a.library,outer_lapse_damping=bad)
  except FloatingPointError:pass
  else:raise AssertionError('invalid damping accepted')
 try:Kernel(a.baseline,outer_lapse_damping=.25)
 except AttributeError:pass
 else:raise AssertionError('old library silently accepted new parameter')
 out=dict(default_bitwise_equal=True,core_bitwise_equal=True,initial_values_unchanged=True,principal_symbol_max_difference=symbol_error,source_balanced_identity_error=error,reference_rhs_difference=stationary,normal_checks=rows,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(Path(a.baseline).read_bytes()).hexdigest())
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
 for k in (old,base,new):k.close()
if __name__=='__main__':main()

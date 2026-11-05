#!/usr/bin/env python3
"""Off-constraint jet/RHS parity and normalized spatial representation checks."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import ChartEvolution,ReferenceChiEvolution
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 k=Kernel(a.library,separated=True,compact_end=.97);r=np.linspace(.001,.999,257);rng=np.random.default_rng(329);v=np.tile([.8,1.1,.02,-.5,.01,.03,.7,-.1,.02,.04],(len(r),1));v+=rng.normal(0,.01,v.shape);d=rng.normal(0,.02,v.shape);dd=rng.normal(0,.03,v.shape);rows=[]
 for power in [2,4]:
  q,D,DD=k.reference_chi(r,v,power,d,dd);expected=v.copy();first=d.copy();second=dd.copy()
  expected[:,0]=r**power*v[:,0];first[:,0]=r**power*d[:,0]+power*r**(power-1)*v[:,0]
  second[:,0]=r**power*dd[:,0]+2*power*r**(power-1)*d[:,0]+power*(power-1)*r**(power-2)*v[:,0]
  full=k.expand_chart(q,D,DD);oracle=k.expand_chart(expected,first,second);_,src,_=k.initial(r)
  left=k.parts(r,*full,src);right=k.parts(r,*oracle,src)
  err=max(float(np.max(abs(x-y)/(1+abs(y)))) for x,y in zip(left[:3],right[:3]));assert err<1e-10,err
  inverse=k.reference_chi(r,q,power,D,DD,inverse=True)
  roundtrip=max(float(np.max(abs(x-y)/(1+abs(y)))) for x,y in zip(inverse,(v,d,dd)))
  # The quotient second derivative can lose digits as r->0; record that scaling.
  assert roundtrip<3e-8,roundtrip
  args=dict(normal_policy='analytic',reference_jets=True,reference_step=.000125,reference_tail='scri')
  direct=ChartEvolution(k,128,**args);scaled=ReferenceChiEvolution(k,128,chi_reference_power=power,**args)
  f=scaled.materialize(scaled.initial);ds,dds=scaled.spatial_jets(f);dc,ddc=direct.spatial_jets(f)
  initial=float(max(np.max(abs(ds-dc)),np.max(abs(dds-ddc))));assert initial<2e-8,initial
  rate=scaled.rhs(scaled.initial);assert max(abs(rate[scaled.r<.4]).flat)>1e-3
  # A fixed linear RK coordinate scaling is exactly the same RHS; no background
  # RHS is subtracted. Physical-chi KO is unchanged by this experiment.
  rows.append(dict(power=power,RHS_scaled_error=err,jet_roundtrip_error=roundtrip,initial_jet_absolute_difference=initial,central_actual_RHS_max=float(np.max(abs(rate[scaled.r<.4])))))
 (a.output/'audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows));k.close()
if __name__=='__main__':main()

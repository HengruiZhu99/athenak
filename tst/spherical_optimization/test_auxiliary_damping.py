#!/usr/bin/env python3
"""An exposed shared parameter changes exactly -kappa_D D, including off constraint."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 base=Kernel(a.library,separated=True,compact_end=.97);r=np.linspace(.001,.999,257);q,src,om=base.initial(r);rng=np.random.default_rng(194)
 q[:,[3,4,5,6,7,10,11]]+=rng.normal(0,.02,(len(r),7));d=rng.normal(0,.02,q.shape);dd=rng.normal(0,.03,q.shape);R,S,T,_=base.parts(r,q,d,dd,src);rows=[]
 for damping in [0.,2.,4.,8.]:
  k=Kernel(a.library,separated=True,compact_end=.97,auxiliary_damping=damping);initial,source,omega=k.initial(r)
  assert np.array_equal(source,src) and np.array_equal(omega,om) and np.array_equal(initial,base.initial(r)[0])
  X,Y,Z,_=k.parts(r,q,d,dd,src);expected=R.copy();expected[:,11]-=(damping-1)*q[:,11]
  error=float(np.max(abs(X-expected)));assert error<2e-12 and np.array_equal(S,Y) and np.array_equal(T,Z)
  supplied=rng.normal(0,.3,len(r));out=np.zeros(len(r));k.check(k.lib.spherical_auxiliary(k.handle,len(r),r,q,d,supplied,out))
  analytic=supplied+q[:,9]*(d[:,11]-d[:,7])-damping*q[:,11];aux=float(np.max(abs(out-analytic)));assert aux<2e-15
  P=base.symbol(.9,q[-30]);Q=k.symbol(.9,q[-30]);assert np.array_equal(P,Q)
  rows.append(dict(damping=damping,RHS_absolute_error=error,actual_auxiliary_absolute_error=aux,principal_symbol_changed=False));k.close()
 for bad in [-1.,float('nan'),float('inf')]:
  try:Kernel(a.library,auxiliary_damping=bad)
  except FloatingPointError:pass
  else:raise AssertionError('invalid damping accepted')
 (a.output/'audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows));base.close()
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent 70-digit inversion/differentiation of the outer CMC map."""
import argparse,json
from pathlib import Path
import numpy as np
import mpmath as mp
from kernel import Kernel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);mp.mp.dps=70
 k=Kernel(a.library,slicing_ratio=4,separated=True);rows=[]
 def fields(r):
  y=mp.findroot(lambda z:mp.quad(lambda t:1/mp.sqrt(1+t*t-mp.mpf(2)/3*t**3),[0,z])+mp.log(r),-mp.log(r))
  o=(1-r*r)/2
  return (r*y/o)**2,o/y*mp.sqrt(1+y*y-mp.mpf(2)/3*y**3)
 for radius in (.97,.98,.99,.995,.999):
  r=mp.mpf(str(radius));d,dd=k.outer_reference_jets(np.array([radius]));q,_,_=k.initial(np.array([radius]));exact=np.array([[float(mp.diff(lambda x:fields(x)[f],r,order)) for f in (0,1)] for order in (1,2)])
  actual=np.array([d[0,[0,8]],dd[0,[0,8]]]);error=np.max(abs(actual-exact));value_error=max(abs(q[0,[0,8]]-np.array([float(v) for v in fields(r)])))
  assert error<3e-7 and value_error<1e-11
  rows.append(dict(r=radius,derivative_error=float(error),value_error=float(value_error),actual=actual.tolist(),exact=exact.tolist()))
 d,dd=k.outer_reference_jets(np.array([1.]));assert d[0,8]==dd[0,8]==1 and d[0,9]==-1 and dd[0,9]==0
 assert np.max(abs(d[0,[0,1,2,3,4,5,6,7,10,11]]))==0 and np.max(abs(dd[0,[0,1,2,3,4,5,6,7,9,10,11]]))==0
 (a.output/'audit.json').write_text(json.dumps(dict(precision_digits=70,mpmath_version=mp.__version__,points=rows,scri_first=d.tolist(),scri_second=dd.tolist(),qualification='Derivatives of the shared native S=1 reference only; evolved RHS unchanged.'),indent=2)+'\n');print(json.dumps(rows));k.close()
if __name__=='__main__':main()

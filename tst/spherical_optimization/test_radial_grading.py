#!/usr/bin/env python3
"""Monotone grid and derivative reproduction across the disclosed grid family."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import ChartEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);k=Kernel(a.library,slicing_ratio=4,separated=True);rows=[]
 for kind in ('quadratic','two-end'):
  for grading in (-.5,-.3,0.,.3,.5):
   e=ChartEvolution(k,64,grading=grading,grid_map=kind);r=e.r;assert r[0]>0 and r[-1]==1 and np.all(np.diff(r)>0)
   # Polynomial derivatives must also hold at the puncture/scri one-sided rows.
   errors=[]
   for degree in range(1,7):
    d,dd=e.op.derivatives(r**degree,1);exactd=degree*r**(degree-1);exactdd=degree*(degree-1)*r**max(0,degree-2)
    errors.append(float(max(np.max(abs(d-exactd)),np.max(abs(dd-exactdd)))))
   assert max(errors)<1e-8
   rows.append(dict(map=kind,grading=grading,min_dr=float(e.h),scri_dr=float(e.dr[-1]),max_polynomial_error=max(errors)))
 (a.output/'audit.json').write_text(json.dumps(dict(grids=rows,qualification='Spatial geometry/stencil consistency only; one weak-penalty two-end evolution is known unstable.'),indent=2)+'\n');print(json.dumps(rows));k.close()
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Algebraic consistency of a REJECTED boundary closure, not a stability test."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 rows=[]
 for width in (5,7):
  cfg=dict(n=64,algebraic_chart=True,chi_reference_power=2,compact_end=.97,separated=True,outflow_boundary=True,outflow_width=width,normal_policy='analytic',reference_jets=True,reference_tail='scri',reference_step=.000125)
  k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);r=e.r;q=e.initial.copy()
  q[:,0]=.8+.1*r;q[:,1]=1+.01*r*r;q[:,2]=.03*r;q[:,3]=-3+.02*r*r;q[:,4]=.01*r
  q[:,5]=.05*r+.02*r*r;q[:,6]=.7+.1*r;q[:,7]=-.1*r
  q[:,9]=q[:,5]+.3*r+.02*r*r-.1*r**3
  e.rhs(q);b=e.last;trace=b['q'][-1,11]-b['q'][-1,7]
  assert abs(trace-.22)<2e-10 and abs(b['clock'])<2e-10
  assert np.array_equal(b['q'][:,6],q[:,4]) and np.array_equal(b['q'][:,7],q[:,5])
  rows.append(dict(width=width,clock=float(b['clock']),outgoing_trace=float(trace),normal_corrections=e.boundary_derivative_correction.tolist(),matrix_condition=e.boundary_matrix_condition));k.close()
 result=dict(rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),qualification='Only clock/null/outgoing trace algebra; frozen operators reject this closure. No differential compatibility or stability assertion.')
 (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

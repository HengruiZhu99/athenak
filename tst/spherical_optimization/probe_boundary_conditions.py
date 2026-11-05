#!/usr/bin/env python3
"""Local rank test, not a candidate boundary implementation or stability proof."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import ChartEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);k=Kernel(a.library,separated=True);e=ChartEvolution(k,128,normal_policy='analytic',reference_jets=True);rows=[]
 for name,state in [('initial',e.initial),('1M',np.load(a.run/'last-accepted.npz')['independent_state'])]:
  e.rhs(state);x=e.last;Q=x['q'][-1];d=x['d'][-1];dd=x['dd'][-1]
  def function(derivative):
   normal=k.analytic_normal(Q,derivative,dd);_,dependent=k.scri(Q,derivative,dd,normal,e.dr[-1],e.penalty)
   return np.array([dependent[1],dependent[0]])
  base=function(d);columns=[]
  for f in (8,9):
   v=d.copy();v[f]+=1;columns.append(function(v)-base)
  A=np.array(columns).T;change=np.linalg.solve(A,np.array([0.,Q[11]])-base)
  v=d.copy();v[[8,9]]+=change;residual=function(v)-np.array([0.,Q[11]])
  assert np.max(abs(residual))<1e-11
  rows.append(dict(state=name,matrix=A.tolist(),condition=float(np.linalg.cond(A)),determinant=float(np.linalg.det(A)),derivative_correction=change.tolist(),residual=residual.tolist()))
 (a.output/'audit.json').write_text(json.dumps(dict(points=rows,qualification='Local derivative conditions only; no auxiliary evolution or boundary well-posedness established.'),indent=2)+'\n');k.close()
if __name__=='__main__':main()

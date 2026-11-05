#!/usr/bin/env python3
"""Verify that the algebraic D input is redundant at every RK stage."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import ChartEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 k=Kernel(a.library,separated=True);e=ChartEvolution(k,64,normal_policy='analytic',reference_jets=True)
 q=e.initial.copy();q[:,9]+=.001*np.exp(-((e.r-.9)/.1)**2);q[-1,9]=33;rate=e.rhs(q);x=e.last.copy()
 p=q.copy();p[-1,9]=-27;other=e.rhs(p);assert np.array_equal(rate,other)
 d,dd=e.op.derivatives(x['q'][:,11],-1)
 assert np.max(abs(d-x['d'][:,11]))<1e-12 and np.max(abs(dd-x['dd'][:,11]))<1e-12
 first=e.step(q,1e-5);second=e.step(p,1e-5);assert np.array_equal(first,second)
 out=dict(rhs_D_input_difference=float(np.max(abs(rate-other))),step_D_input_difference=float(np.max(abs(first-second))),centered_D_jet_error=float(max(np.max(abs(d-x['d'][:,11])),np.max(abs(dd-x['dd'][:,11])))),qualification='D is already materialized before its auxiliary advection at each stage. This does not assert continuum differential trace compatibility.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()

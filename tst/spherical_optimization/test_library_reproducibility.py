#!/usr/bin/env python3
"""Compare compiled libraries on initial data and nontrivial off-constraint jets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--baseline',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 old=Kernel(a.baseline,slicing_ratio=4,separated=True);new=Kernel(a.library,slicing_ratio=4,separated=True);r=np.linspace(.003,.999,257);q,src,om=old.initial(r);Q,S,O=new.initial(r)
 rng=np.random.default_rng(732);q[:,[3,4,5,6,7,10,11]]+=rng.normal(0,.01,(len(r),7));d=rng.normal(0,.03,q.shape);dd=rng.normal(0,.05,q.shape)
 x=old.parts(r,q,d,dd,src);y=new.parts(r,q,d,dd,src);errors=[]
 for left,right in zip(x[:3],y[:3]):errors.append(float(np.max(abs(left-right)/(1+abs(left)))))
 assert max(errors)<1e-11 and np.max(abs(Q-old.initial(r)[0]))<1e-11
 out=dict(initial_max_difference=float(np.max(abs(Q-old.initial(r)[0]))),source_max_difference=float(np.max(abs(S-src))),normalized_RHS_errors=errors,baseline_sha256=hashlib.sha256(Path(a.baseline).read_bytes()).hexdigest(),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),qualification='Compiled-kernel parity on 257 nontrivial jets; optimization speedup is separate from formulation comparisons.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));old.close();new.close()
if __name__=='__main__':main()

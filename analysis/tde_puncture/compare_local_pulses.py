#!/usr/bin/env python3
"""Check linear amplitude scaling of measured maxima; not a fieldwise norm."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_runs import history

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('single',type=Path);p.add_argument('double',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
x,_=history(a.single/'ks_background.user.hst')
y,_=history(a.double/'ks_background.user.hst')
samples=[]
for t,q in zip(y['time'],y['Theta-max']):
    hits=np.flatnonzero(abs(x['time']-t)<1e-9)
    if len(hits)==1 and x['Theta-max'][hits[0]]>0:
        samples.append((float(t),float(q/(2*x['Theta-max'][hits[0]]))))
assert samples
result={'quantity':'max Theta(2 epsilon)/(2 max Theta(epsilon)); maxima may occupy different cells',
        'samples':samples,'windows':{}}
for lo,hi in [(0,30),(30,60),(60,100)]:
    v=np.array([q for t,q in samples if lo<=t<=hi])
    if len(v):result['windows'][f'{lo}_{hi}']={'samples':len(v),
        'min':float(v.min()),'max':float(v.max()),'median':float(np.median(v))}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['windows'],indent=2))

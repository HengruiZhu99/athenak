#!/usr/bin/env python3
"""Retain component-wise orders/alignment; aggregate convergence can hide these."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();z=np.load(a.input)
d1=z['coarse_minus_medium'].reshape(50,-1);d2=z['medium_minus_fine'].reshape(50,-1)
records=[]
for n in range(50):
    n1=float(np.sqrt(np.sum(d1[n]*d1[n])));n2=float(np.sqrt(np.sum(d2[n]*d2[n])))
    order=float(np.log2(n1/n2)) if n1>0 and n2>0 else None
    alignment=float(np.sum(d1[n]*d2[n])/(n1*n2)) if n1>0 and n2>0 else None
    record=dict(index=n,difference_L2=[n1,n2],observed_order=order,alignment=alignment)
    if 'temporal_control' in z:
        record['temporal_control_ratio']=float(np.sqrt(np.sum(z['temporal_control'][n]**2))/n2) if n2>0 else None
    records.append(record)
a.output.write_text(json.dumps(dict(input=str(a.input),sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),components=records,scope='component audit; no automatic promotion from aggregate norms'),indent=2)+'\n')
print('minimum alignment',min(r['alignment'] for r in records if r['alignment'] is not None))

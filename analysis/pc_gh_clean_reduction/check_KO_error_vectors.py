#!/usr/bin/env python3
"""Fixed order-six/order-seven prediction of paired spatial error vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--with-ko',type=Path,required=True);p.add_argument('--without-ko',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
with_ko=np.load(a.with_ko);without=np.load(a.without_ko)
d1=with_ko['coarse_minus_medium'];d2=with_ko['medium_minus_fine']
f1=without['coarse_minus_medium'];f2=without['medium_minus_fine']
k1=d1-f1;k2=d2-f2;predicted=64*f2+128*k2
records=[]
norm=lambda x:float(np.sqrt(np.sum(x*x)))
for n in range(50):
    records.append(dict(index=n,relative_residual=norm(predicted[n]-d1[n])/norm(d1[n]),
        prediction_alignment=float(np.sum(predicted[n]*d1[n])/(norm(predicted[n])*norm(d1[n]))),
        KO_response_order=float(np.log2(norm(k1[n])/norm(k2[n]))),
        KO_response_alignment=float(np.sum(k1[n]*k2[n])/(norm(k1[n])*norm(k2[n])))))
rho=records[1];passed=rho['relative_residual']<=.1 and rho['prediction_alignment']>=.99 and abs(rho['KO_response_order']-7)<=.3
result=dict(status='PASS' if passed else 'FAIL',rho=rho,components=records,
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.with_ko,a.without_ko]},
    scope='fixed-coefficient prediction from matched solution errors; no fitted exponent or raw stage source attribution')
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez(a.output/'signed-vectors.npz',FD_coarse=f1,FD_fine=f2,KO_response_coarse=k1,KO_response_fine=k2,prediction=predicted,residual=predicted-d1)
print(json.dumps(result['rho']));assert passed

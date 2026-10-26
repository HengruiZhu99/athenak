#!/usr/bin/env python3
from decimal import Decimal,localcontext
from pathlib import Path
import json,argparse,hashlib
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();E=a.evidence;records=[]
def compute(x):
 one=x[0]*0+1;two=one*2;r2=sum(t*t for t in x);Om=(one-r2)/two;N=one/Om-one;gamma=one/Om**2;sig=[t/Om for t in x];sig2=[[(one if i==j else one*0)/Om+x[i]*x[j]/Om**2 for j in range(3)] for i in range(3)];lap=sum(sig2[i][i] for i in range(3));grad=sum(t*t for t in sig);nd=[t/Om**2 for t in x];out=[]
 for i in range(3):
  for j in range(3):
   delta=one if i==j else one*0;ricci=-sig2[i][j]-delta*lap+sig[i]*sig[j]-delta*grad;hess=delta/Om**2+two*x[i]*x[j]/Om**3
   for k in range(3):hess-=((sig[j] if k==i else one*0)+(sig[i] if k==j else one*0)-(sig[k] if i==j else one*0))*nd[k]
   lie=two*r2*delta/Om**3+two*delta/Om**2;out.append(-hess+N*(ricci+delta*gamma)+lie) # K*Kij-2K_i^kK_kj = gammaij.
 return max(abs(z) for z in out)
for z in map(json.loads,(E/'extended-rhs-jets.jsonl').read_text().splitlines()):
 if z['family']!=2:continue
 with localcontext() as ctx:
  ctx.prec=60;d=compute([Decimal(str(t)) for t in z['coordinate']])
 f=compute([float(t) for t in z['coordinate']]);records.append(dict(radius=z['radius'],direction=z['direction'],kappa1=z['kappa1'],kappa2=z['kappa2'],decimal_absolute_source=str(d),float_absolute_source=f,passed=d<=Decimal('1e-40')))
r=dict(passed=all(z['passed'] for z in records),records=records,max_decimal=max(float(z['decimal_absolute_source']) for z in records),max_float=max(z['float_absolute_source'] for z in records),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Independent analytic Minkowski ADM cancellation only. Not higher-precision native template evaluation; original source gate failures retained.');(E/'minkowski-decimal-review.json').write_text(json.dumps(r,indent=2)+'\n');print(r['passed'],r['max_decimal'],r['max_float'])

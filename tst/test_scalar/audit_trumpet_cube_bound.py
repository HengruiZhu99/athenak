#!/usr/bin/env python3
"""Sufficient all-face outflow proof for an axis-aligned inner trumpet cube."""
import argparse,hashlib,json,math
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing characteristic proof is immutable')
records=[]
for spin,n,cut in [(0,n,.25) for n in [24,36,48,60]]+[(0,60,.20)]+[(.5,n,.126) for n in [24,32,40]]+[(.5,n,.06) for n in [36,40,48,60]]+[(.9,60,.06)]:
 h=3/n;removed=sum((j+.5)*h<cut for j in range(n));d=removed*h;c=math.sqrt(1-spin*spin);limit=(c-abs(spin))/3;bound=((c-abs(spin))**2/3-3*d*d)/((1+math.sqrt(3)*d)**2+spin*spin) if c>abs(spin) else None
 records.append(dict(spin=spin,n=n,requested_cut=cut,actual_half_width=d,maximum_sufficient_half_width=limit,has_cut=bool(removed),strict_outflow_proven=bool(removed and 0<d<limit),negative_inverse_normal_lower_bound=bound if removed and 0<d<limit else None))
result=dict(scope='Exact physical inner trumpet only: Omega=1 and hprime=0 on the complete cube. No reference metric, small-spin expansion or time evolution.',derivation='For n=x/r and z=(-n_y,n_x,0), g^{ij}=[r^2 delta_ij-(c n_i-a z_i)(c n_j-a z_j)]/Sigma. On a face x_A=sign*d, sign*(c n_A-a z_A)>=d*(c-|a|)/r. Since r^2<=3d^2, d<(c-|a|)/3 makes g^{AA}<0 everywhere. With A=(r+1)^2+a^2=B+r^2 and B=2r+1+a^2>0, sign*beta^A=r[B*sign*(c n_A-a z_A)+c*r^2*sign*n_A]/X>0. Thus b=-beta.normal>0 and both physical speeds b +/- sqrt(b^2-g^{AA}/g^{tt}) are strictly positive on every inward face, edge and corner. This is sufficient, not a claim that every cube failing this bound is invalid.',records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records,indent=2))

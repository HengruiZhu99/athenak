#!/usr/bin/env python3
"""Exact symbolic principal cancellation, after physical metric raising.
Coefficients are frozen only for principal extraction. Lower sources remain in
full nonlinear chains and are not asserted to vanish by this check.
"""
from fractions import Fraction
from collections import defaultdict
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
criteria={'status':'Predeclared before coefficient check','gate':'Exact zero coefficients of H-gradient, M-divergence and Z longitudinal second derivatives after closed H/M substitution; remaining principal is N^2 gammaInv^ij partial_i partial_j for Theta and each Zup component.','limits':'Formal principal extraction only, no lower-source energy/SAT validation.'}
(out/'wave-cancellation-criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
def add(p,c,*f):p[tuple(sorted(f))]+=Fraction(c)
def clean(p):return {str(k):str(v) for k,v in p.items() if v}
t=defaultdict(Fraction)
for i in range(3):
 for j in range(3):
  g=f'G{min(i,j)}{max(i,j)}'
  add(t,-1,'N','N',g,f'M{j},d{i}') # N/2 D0H
  add(t,1,'N','N',g,f'M{j},d{i}') # N div(N Fup)
  add(t,1,'N','N',g,f'Theta,d{min(i,j)}{max(i,j)}')
  add(t,-1,'N','N',g,f'Theta,d{min(i,j)}{max(i,j)}') # subtract desired wave
zs=[]
for a in range(3):
 z=defaultdict(Fraction)
 for j in range(3):
  g=f'G{min(a,j)}{max(a,j)}'
  add(z,Fraction(-1,2),'N','N',g,f'H,d{j}')
  add(z,Fraction(1,2),'N','N',g,f'H,d{j}')
  add(z,-1,'N','N',g,f'divZ,d{j}')
  add(z,1,'N','N',g,f'divZ,d{j}')
  for k in range(3):
   gg=f'G{min(j,k)}{max(j,k)}';zz=f'Z{a},d{min(j,k)}{max(j,k)}'
   add(z,1,'N','N',gg,zz);add(z,-1,'N','N',gg,zz)
 zs.append(clean(z))
review={'theta_remainder':clean(t),'Zup_remainders':zs,'exact_principal_cancellation_pass':not clean(t) and not any(zs),'lightcone':'(lambda-beta.n)^2 = N^2 gammaInv(n,n) = alpha^2 chi gtildeInv(n,n)','lower_sources_checked_here':False}
(out/'wave-cancellation-review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))

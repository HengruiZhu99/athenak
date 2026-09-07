#!/usr/bin/env python3
"""Exact arbitrary-function reduction, curl, Cartan and chart-curl identities."""
import argparse
import json
from pathlib import Path
import sympy as s
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
x=s.symbols('x:3');f=s.Function('F')(*x);u=s.Function('u')(*x)
b=[s.Function('b'+str(i))(*x) for i in range(3)]
G=[s.Function('G'+str(i))(*x) for i in range(3)]
lam=s.Function('lambda')(*x);E=[G[i]-s.diff(u,x[i]) for i in range(3)]
ut=sum(b[j]*s.diff(u,x[j]) for j in range(3))+f
Gt=[sum(b[j]*s.diff(G[i],x[j])+s.diff(b[j],x[i])*G[j] for j in range(3))+s.diff(f,x[i])-lam*E[i] for i in range(3)]
om=[[s.diff(G[j],x[i])-s.diff(G[i],x[j]) for j in range(3)] for i in range(3)]
for i in range(3):
 expected=sum(b[j]*s.diff(E[i],x[j])+s.diff(b[j],x[i])*E[j] for j in range(3))-lam*E[i]
 assert s.expand(Gt[i]-s.diff(ut,x[i])-expected)==0
 cartan=s.diff(f+sum(b[j]*G[j] for j in range(3)),x[i])+sum(b[j]*om[j][i] for j in range(3))-lam*E[i]
 assert s.expand(Gt[i]-cartan)==0
for i,j in [(0,1),(0,2),(1,2)]:
 expected=sum(b[k]*s.diff(om[i][j],x[k])+s.diff(b[k],x[i])*om[k][j]+s.diff(b[k],x[j])*om[i][k] for k in range(3))-lam*om[i][j]-s.diff(lam,x[i])*E[j]+s.diff(lam,x[j])*E[i]
 assert s.expand(s.diff(Gt[j],x[i])-s.diff(Gt[i],x[j])-expected)==0
# An arbitrary metric component g(s) suffices: tensor indices are spectators.
chart=[s.Function('s'+str(i))(*x) for i in range(5)]
g=s.Function('metric_component')(*chart)
S=[[s.Function('S%d%d'%(i,A))(*x) for A in range(5)] for i in range(3)]
J=[s.diff(g,v) for v in chart];H=[[s.diff(J[A],chart[B]) for B in range(5)] for A in range(5)]
Q=[sum(J[A]*S[i][A] for A in range(5)) for i in range(3)]
ES=[[S[i][A]-s.diff(chart[A],x[i]) for A in range(5)] for i in range(3)]
for i,j in [(0,1),(0,2),(1,2)]:
 target=sum(J[A]*(s.diff(S[j][A],x[i])-s.diff(S[i][A],x[j])) for A in range(5))
 target+=sum(H[A][B]*(s.diff(chart[B],x[i])*ES[j][A]-s.diff(chart[B],x[j])*ES[i][A]) for A in range(5) for B in range(5))
 assert s.expand(s.diff(Q[j],x[i])-s.diff(Q[i],x[j])-target)==0
out=dict(status='PASS',arbitrary_potential_reduction_components=3,arbitrary_potential_curl_pairs=3,cartan_components=3,arbitrary_five_chart_metric_curl_pairs=3,scope='exact arbitrary smooth function identities; no discrete commutation or puncture regularity claim')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

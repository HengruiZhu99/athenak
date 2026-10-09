from pathlib import Path
import json,time,sympy as s
from sympy.polys.matrices import DomainMatrix
P=Path(__file__).resolve().parent;j=json.loads((P/'pilot-actual.json').read_text());cols=j['orientations'][0]['columns']
rat=lambda x:s.Rational(str(x)).limit_denominator(1000000)
E=s.Matrix([c['E']for c in cols]).T.applyfunc(rat)
L=s.Matrix([[rat(c['N1_t'])for c in cols]]);R=s.Matrix([[rat(c['delta_Rq'])for c in cols]])
N=s.Matrix([[rat(c['N0_t'])for c in cols]]);Q=s.Matrix([[rat(c['Q0_t'])for c in cols]])
B=s.Matrix([c['next_R0']for c in cols]).T.applyfunc(rat)
t=time.monotonic();dm=DomainMatrix.from_Matrix(E).convert_to(s.QQ);rref,pivots=dm.rref();er=rref.to_Matrix();print('rankE',len(pivots),'seconds',time.monotonic()-t,flush=True)
def remainder(v):
 v=v.copy()
 for i,p in enumerate(pivots):v-=v[:,p]*er[i,:]
 return v
out={'rankE':len(pivots),'N0_remainder_nonzeros':sum(x!=0 for x in remainder(N)),'Q0_remainder_nonzeros':sum(x!=0 for x in remainder(Q)),'L_plus_Rq_over_a2_nonzeros':sum(x!=0 for x in remainder(L+4*R)),'L_remainder_nonzeros':sum(x!=0 for x in remainder(L)),'Rq_remainder_nonzeros':sum(x!=0 for x in remainder(R)),'next_R0_remainder_nonzeros':[sum(x!=0 for x in remainder(B[i,:]))for i in range(20)]}
print(out,flush=True)
for name,v in [('LplusR',L+4*R),('N0',N),('Q0',Q)]:
 rv=remainder(v)
 out[name+'_remainder']=[[i,str(x)]for i,x in enumerate(rv)if x]
(P/'pilot-rowspace.json').write_text(json.dumps(out,indent=2)+'\n')

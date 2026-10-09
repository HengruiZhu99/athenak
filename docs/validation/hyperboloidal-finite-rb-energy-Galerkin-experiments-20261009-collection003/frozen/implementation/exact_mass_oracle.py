"""Exact common-rho polynomial norm oracle, no PDE assembly."""
from pathlib import Path
import hashlib,json,sympy as s,time
P=Path(__file__).resolve().parent;d=P/'polynomial-mass-gate-001';assert not d.exists();d.mkdir()
plan={'N':8,'L':list(range(5)),'rho_domain':[0,.98**2],'exact_dimensionless_weight':'1/2 t^(L+1/2), t=rho/rb^2','exact_target':'delta_ij/[2(2i+L+3/2)]','numerical_scaled_tolerance':5e-11,'scope':'Polynomial trial/mass/normalization only; no radial PDE or boundary operator'}
(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');start=time.monotonic();t=s.symbols('t');rows=[]
for L in range(5):
 polys=[s.Poly(s.jacobi(k,0,s.Rational(2*L+1,2),2*t-1),t) for k in range(8)]
 maximum=s.Rational(0)
 for i,pi in enumerate(polys):
  for j,pj in enumerate(polys):
   q=s.Poly(pi.as_expr()*pj.as_expr(),t);integral=sum(v/(2*(k[0]+s.Rational(2*L+3,2))) for k,v in q.terms());target=1/(2*(2*i+s.Rational(2*L+3,2))) if i==j else s.Rational(0)
   assert s.simplify(integral-target)==0,(L,i,j,integral,target)
 rows.append({'L':L,'polynomials_ascending':[[str(pi.nth(k)) for k in range(8)] for pi in polys],'exact_diagonal':[str(1/(2*(2*i+s.Rational(2*L+3,2)))) for i in range(8)],'exact_all_64_entries_pass':True})
r={'passed_exact':True,'rows':rows,'elapsed':time.monotonic()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sympy':s.__version__};(d/'exact.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))

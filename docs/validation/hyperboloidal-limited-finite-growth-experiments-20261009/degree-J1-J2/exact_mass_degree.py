"""Exact polynomial mass extension at released N12/N16; no PDE action."""
from pathlib import Path
import argparse,hashlib,json,time
import sympy as s
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--N',type=int,required=True);a=p.parse_args();N=a.N
assert N in (12,16);d=P/f'polynomial-mass-N{N}';assert not d.exists();d.mkdir()
plan={'N':N,'L':list(range(5)),'rho_domain':[0,.98**2],
      'exact_dimensionless_weight':'1/2 t^(L+1/2), t=rho/rb^2',
      'exact_target':'delta_ij/[2(2i+L+3/2)]','numerical_scaled_tolerance':5e-11,
      'method':'independent exact rational polynomial coefficient products/integrals',
      'scope':'polynomial trial/mass/normalization only; no radial PDE or boundary operator'}
(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
start=time.monotonic();t=s.symbols('t');rows=[];count=0
for L in range(5):
    polys=[s.Poly(s.jacobi(k,0,s.Rational(2*L+1,2),2*t-1),t) for k in range(N)]
    coeffs=[[pi.nth(k) for k in range(N)] for pi in polys]
    for i in range(N):
        for j in range(N):
            integral=s.Rational(0)
            for ki in range(i+1):
                for kj in range(j+1):
                    integral+=coeffs[i][ki]*coeffs[j][kj]/(2*(ki+kj+s.Rational(2*L+3,2)))
            target=1/(2*(2*i+s.Rational(2*L+3,2))) if i==j else s.Rational(0)
            assert integral==target,(L,i,j,integral,target);count+=1
    rows.append({'L':L,'polynomials_ascending':[[str(v) for v in c] for c in coeffs],
                 'exact_diagonal':[str(1/(2*(2*i+s.Rational(2*L+3,2)))) for i in range(N)],
                 'exact_all_entries_pass':True,'entries':N*N})
result={'passed_exact':True,'N':N,'entries':count,'rows':rows,'elapsed':time.monotonic()-start,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sympy':s.__version__}
(d/'exact.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

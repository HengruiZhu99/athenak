"""Exact rational harmonic principal proof + actual792 matrix readback. No eigs."""
import json,math,pathlib,sys
import sympy as s
R=s.Rational
S=s.Matrix([[0,0,0,-1,0,0,0,0],[0,0,0,R(2,3),R(4,3),0,0,R(-2,3)],[0,0,0,0,0,-2,0,R(4,3)],[-1,0,0,0,0,0,0,0],[0,1,0,0,0,0,R(1,2),0],[R(-2,3),R(1,3),R(-1,2),0,0,0,R(2,3),0],[0,0,0,R(-4,3),R(-2,3),0,0,R(4,3)],[-1,R(1,2),0,0,0,0,1,0]])
V=s.Matrix([[0,-2,0,1],[R(-1,2),0,R(1,2),0],[0,0,0,1],[0,0,1,0]])
T=s.Matrix([[0,-2],[R(-1,2),0]])
M=s.diag(S,V,V,T,T);I=s.eye(20);pp=(I+M)/2;pm=(I-M)/2;H=(I+M.T*M)/2
assert M*M==I and s.trace(M)==0
assert pp*pp==pp and pm*pm==pm and pp*pm==s.zeros(20) and pp+pm==I
assert pp.rank()==pm.rank()==10 and H*M==M.T*H
assert H==pp.T*pp+pm.T*pm
expected=[[float(M[i,j])for j in range(20)]for i in range(20)]
# Deterministic scalar loops avoid BLAS warning behavior and require no numpy.
def mul(a,b):return [[math.fsum(a[i][k]*b[k][j]for k in range(20))for j in range(20)]for i in range(20)]
def transpose(a):return [list(x)for x in zip(*a)]
rows=json.loads(pathlib.Path(sys.argv[1]).read_text());assert len(rows)==792
err=inv=sym=normal=0
for q in rows:
 a=q['M'];assert len(a)==20 and all(len(r)==20 for r in a)and all(math.isfinite(x)for r in a for x in r)
 err=max(err,max(abs(a[i][j]-expected[i][j])for i in range(20)for j in range(20)))
 aa=mul(a,a);inv=max(inv,max(abs(aa[i][j]-(i==j))for i in range(20)for j in range(20)))
 at=transpose(a);ata=mul(at,a);h=[[(ata[i][j]+(i==j))/2 for j in range(20)]for i in range(20)]
 hm=mul(h,a);mth=mul(at,h);sym=max(sym,max(abs(hm[i][j]-mth[i][j])for i in range(20)for j in range(20)))
 normal=max(normal,q['normal_scaled'])
assert err<=2e-12,(err,'expected');assert inv<=2e-11,(inv,'involution');assert sym<=2e-11,(sym,'symmetrizer');assert normal<=5e-11,(normal,'normal')
print(json.dumps({'passed':True,'actual_principal_cases':len(rows),'matrix_expected_max_absolute':err,'actual_involution_max_absolute':inv,'actual_symmetrizer_max_absolute':sym,'normal_max_scaled':normal,'exact_rational_M2_I':True,'exact_projector_ranks':[10,10],'exact_H_identity_and_symmetry':True,'exact_H_quadratic_lower_bound':'x^T H x=(||x||²+||Mx||²)/2 >= ||x||²/2','raw22_extension_or_lower_order_spectrum_claim':False},indent=2))

#!/usr/bin/env python3
"""Derive a nonlinear regularity chart; does not set H, M, Theta or Z to zero.

Optional derivation dependency: SymPy 1.14 (install in an isolated target).
Production formulas are recorded separately; no numerical evolution is claimed.
"""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--sympy-path',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.sympy_path:sys.path.insert(0,str(a.sympy_path))
import sympy as s
x=s.symbols('x');c=s.symbols('c0:6');P0,P1,E0,E1,pfree,afree=s.symbols('P0 P1 E0 E1 pfree afree')
C=sum(c[i]*x**i for i in range(6));chi=1+x*C
# Initial series expressions use independent constants; derive required boundary
# values. The final chart promotes those values to differential expressions C(x).
K=-3-2*x*C+x*x*P0+x**3*P1+x**4*pfree
B=s.Rational(2,3)*C+x*E0+x*x*E1+x**3*afree
r2=1-2*x;Arr=r2*B
H=(s.Rational(2,3)*K*K-6*chi-2*x*r2*s.diff(chi,x)
 +x*x*(2*r2*s.diff(chi,x,2)-6*s.diff(chi,x)-s.Rational(5,2)*r2*s.diff(chi,x)**2/chi-s.Rational(3,2)*Arr*Arr))
M=-x*r2*s.diff(B,x)+(2+x)*B+s.Rational(3,2)*x*r2*B*s.diff(chi,x)/chi+s.Rational(2,3)*s.diff(K,x)
Hs=s.series(H,x,0,5).removeO().expand();Ms=s.series(M,x,0,4).removeO().expand()
leading=s.solve([Hs.coeff(x,2),Ms.coeff(x,1)],(P0,E0),dict=True)[0]
next_values=s.solve([Hs.coeff(x,3).subs(leading),Ms.coeff(x,2).subs(leading)],(P1,E1),dict=True)[0]
# Promoting P0,E0 to functions adds their derivatives to the next coefficients.
# Derive afresh with the functional expressions, rather than substituting constants.
P0f=s.diff(C,x)/2-C/2-C*C/8
E0f=4*s.diff(C,x)/3-5*C*C/6
Kf=-3-2*x*C+x*x*P0f+x**3*P1+x**4*pfree
Bf=s.Rational(2,3)*C+x*E0f+x*x*E1+x**3*afree
Af=r2*Bf
Hf=s.Rational(2,3)*Kf*Kf-6*chi-2*x*r2*s.diff(chi,x)+x*x*(2*r2*s.diff(chi,x,2)-6*s.diff(chi,x)-s.Rational(5,2)*r2*s.diff(chi,x)**2/chi-s.Rational(3,2)*Af*Af)
Mf=-x*r2*s.diff(Bf,x)+(2+x)*Bf+s.Rational(3,2)*x*r2*Bf*s.diff(chi,x)/chi+s.Rational(2,3)*s.diff(Kf,x)
h3=s.expand(s.series(Hf,x,0,4).removeO()).coeff(x,3)
m2=s.expand(s.series(Mf,x,0,3).removeO()).coeff(x,2)
next_functional=s.solve([h3,m2],(P1,E1),dict=True)[0]
# c1=C_x(0), c2=C_xx(0)/2. Promote exactly these values.
Cf=s.Function('C')(x)
promote={c[0]:Cf,c[1]:s.diff(Cf,x),c[2]:s.diff(Cf,x,2)/2}
print('functional_corner_solution',next_functional,flush=True)
P1f=s.factor(next_functional[P1].subs(promote));E1f=s.factor(next_functional.get(E1,E1).subs(promote))
# Verify full nonlinear divisibility using polynomial test C and arbitrary free
# amplitudes; rational expressions retain chi denominators, never small-O fits.
p1expr=P1f.subs(Cf,C).doit();e1expr=E1f.subs(Cf,C).doit()
Kverified=Kf.subs(P1,p1expr);Bverified=Bf.subs(E1,e1expr)
Hv=Hf.subs({P1:p1expr,E1:e1expr});Mv=Mf.subs({P1:p1expr,E1:e1expr})
Hnumer=s.Poly(s.cancel(Hv*chi).expand(),x);Mnumer=s.Poly(s.cancel(Mv*chi).expand(),x)
assert all(Hnumer.nth(i)==0 for i in range(4))
assert all(Mnumer.nth(i)==0 for i in range(3))
H4=s.factor(Hnumer.nth(4));M3=s.factor(Mnumer.nth(3))
J=s.Matrix([H4,M3]).jacobian([pfree,afree]);det=s.factor(J.det());assert det!=0
record={'status':'spatial nonlinear compatibility chart derived and verified; time evolution and leading subsidiary corner relations not yet derived',
 'assumptions':'g=1, S=a=1; x=Omega; K=p+2Theta; B=A_rr/r^2; chi=1+x C',
 'P0':'C_x/2-C/2-C^2/8','E0':'4 C_x/3-5 C^2/6',
 'P1':str(P1f),'E1':str(E1f),'E1_interpretation':'Free A_free(0); afree is A_free_x(0). The displayed linear polynomial represents its boundary jet, not a frozen function.',
 'chart_K':'-3-2x C+x^2 P0+x^3 P1+x^4 pfree',
 'chart_Arr':'(1-2x)[2C/3+x E0+x^2 E1+x^3 afree]',
 'factorization':'H=x^4 H_regular; M_r/r=x^3 M_regular; exact rational divisibility',
 'H4':str(H4),'M3':str(M3),'constraint_amplitude_jacobian':str(J),'constraint_amplitude_jacobian_determinant':str(det),
 'free_constraints':'Nonzero pfree,afree give independent H4,M3. Theta=x^4 tau and Lambda=x^3 ell retain scalar and connection errors. No H/M/Theta/Z projection.',
 'remaining':'Leading outgoing subsidiary corner amplitudes may be correlated; spatial weights alone are not a closed temporal phase. Derive full transformed RHS and endpoint equations before evolution.',
 'sympy_version':s.__version__}
a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

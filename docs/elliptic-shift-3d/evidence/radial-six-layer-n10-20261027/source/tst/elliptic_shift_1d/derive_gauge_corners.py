#!/usr/bin/env python3
"""Nonlinear gauge-corner audit in x=Omega; not a time-closed evolution chart."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--symbolic-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.symbolic_path));import sympy as s
from kernel import Kernel
import g1_rhs
x=s.symbols('x');C0,C1,C2,L1,L2,Afree,Pfree=s.symbols('C0 C1 C2 L1 L2 Afree Pfree')
C=C0+C1*x+C2*x*x;L=C0/2+L1*x+L2*x*x;c=x*C;u=x*L
alpha=1-x+u;chi=1+c;r2=1-2*x
P0=s.diff(C,x)/2-C/2-C*C/8
P1=(9*C**3+18*C*C-18*C*s.diff(C,x)-20*s.diff(C,x)+4*s.diff(C,x,2))/8
E0=4*s.diff(C,x)/3-5*C*C/6
K=-3-2*x*C+x*x*P0+x**3*P1+x**4*Pfree
B=2*C/3+x*E0+x*x*Afree
# beta/r=-1+z, z_x=-(3/2)alpha B; outer null/clock set z(0)=0.
z=s.integrate(-s.Rational(3,2)*s.expand(alpha*B),x);z-=z.subs(x,0)
F=alpha*alpha+4*x*x*alpha
ct=s.expand(r2*(1-z)*s.diff(c,x)-2*chi*z-alpha*chi*r2*B
    +2*chi/3*(alpha*(K+3)-3*u-3*r2*z)/x)
at=s.expand(r2*(1-z)*s.diff(u,x)+r2*z
    +(-F*(K+3)+r2*(alpha*z-u)-(alpha+1-x)*u)/x)
assert ct.coeff(x,0)==at.coeff(x,0)==0
G=s.factor(ct.coeff(x,1)-2*at.coeff(x,1))
required_C1=s.solve(G,C1)[0]
tangency=s.factor((ct.coeff(x,2)-(1+s.Rational(11,2)*C0)*ct.coeff(x,1)).subs(C1,required_C1))
required_C2=s.solve(tangency,C2)[0]
assert s.expand(required_C1-C0-s.Rational(11,4)*C0*C0)==0

# Independent native tensor evaluation, using exact polynomial shift integration.
parameters={C0:.02,C1:-.013,C2:.007,L1:-.004,L2:.003,Afree:.001,Pfree:-.002}
xx=np.linspace(.02,0,61);r=np.sqrt(1-2*xx);q=np.zeros((len(r),8));d=q.copy();dd=q.copy()
for field,expr in [(0,c),(2,r2*B),(3,K+3),(6,u)]:
    f=s.lambdify(x,expr.subs(parameters),'numpy');fx=s.lambdify(x,s.diff(expr,x).subs(parameters),'numpy');fxx=s.lambdify(x,s.diff(expr,x,2).subs(parameters),'numpy')
    q[:,field]=f(xx);d[:,field]=-r*fx(xx);dd[:,field]=r*r*fxx(xx)-fx(xx)
zz=np.asarray(s.lambdify(x,z.subs(parameters),'numpy')(xx))
ops=SimpleNamespace(r=r,integral=lambda _:zz.copy())
R,S,_,beta,db,ddb=g1_rhs.parts(ops,q,d,dd,restoring=1)
nativeR,nativeS,_=Kernel(a.library).parts(r,q,d,dd,beta,db,ddb,lapse_restoring=1)
expected=np.c_[s.lambdify(x,ct.subs(parameters),'numpy')(xx[:-1]),s.lambdify(x,at.subs(parameters),'numpy')(xx[:-1])]
native=(nativeR[:-1]+nativeS[:-1]/xx[:-1,None])[:,[0,6]]
error=float(np.max(abs(native-expected)));assert error<2e-10,error
record=dict(x='Omega',scope='Necessary nonlinear gauge corner hierarchy only; no closed chart or nonlinear stability claim',
    assumptions='g=1,xi=1,slicing=1,spatial H=Omega^4/M=Omega^3 chart,Theta omitted only beyond displayed low orders',
    C0_time=str(s.factor(ct.coeff(x,1))),L0_time=str(s.factor(at.coeff(x,1))),
    first_tangency=str(G),required_C1=str(required_C1),second_tangency=str(tangency),required_C2=str(required_C2),
    Schwarzschild_mass_example='C0=C1=L1=0,C2=-M/2 requires L2=-3M/4, i.e. delta-alpha=-3M Omega^3/4+...',
    native_tensor_max_error=error,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
